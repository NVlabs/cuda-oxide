/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ordinary device global static example.
//!
//! Build and run with:
//!   cargo oxide run device_global

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::kernel;
use cuda_host::cuda_module;

static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];

#[inline(never)]
fn get_static_weights() -> &'static [[f32; 2]; 4] {
    &STATIC_WEIGHTS
}

#[inline(always)]
unsafe fn load_pair(ptr: *const f32, i_pair: usize) -> [f32; 2] {
    unsafe { *(ptr as *const [f32; 2]).add(i_pair) }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to a writable `u64` in device-accessible memory.
    /// The static globals `DEVICE_COUNTER` and `DEVICE_MARKER` are mutated
    /// without synchronisation; the test launches a single thread to dodge
    /// the race.
    #[kernel]
    pub unsafe fn device_global(out: *mut u64) {
        unsafe {
            DEVICE_COUNTER += 1;
            DEVICE_MARKER = 0x00C0_FFEE;
            *out = DEVICE_COUNTER ^ (DEVICE_MARKER as u64);
        }
    }

    /// Read a non-zero immutable Rust static through a flattened pointer.
    ///
    /// This mirrors generated coefficient tables that return
    /// `&'static [[f32; 2]; N]`, then vector-load from `&table[0][0]`.
    #[kernel]
    pub unsafe fn nonzero_static_table(out: *mut f32) {
        let weights = get_static_weights();
        let pair = unsafe { load_pair(&weights[0][0], 2) };
        unsafe {
            *out = pair[0] + pair[1];
        }
    }
}

fn main() {
    println!("=== Device Global Static Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate output");

    let module = ctx
        .load_module_from_file("device_global.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    for launch_idx in 1..=2 {
        unsafe {
            module.device_global(
                &stream,
                LaunchConfig::for_num_elems(1),
                out_dev.cu_deviceptr() as *mut u64,
            )
        }
        .expect("Kernel launch failed");

        let result = out_dev.to_host_vec(&stream).expect("Failed to copy result")[0];
        let expected = launch_idx ^ 0x00C0_FFEEu64;

        println!("Launch {launch_idx}: result = {result:#x}");
        if result != expected {
            eprintln!("FAILED: expected {expected:#x}, got {result:#x}");
            std::process::exit(1);
        }
    }

    let static_out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, 1).expect("Failed to allocate static output");
    unsafe {
        module.nonzero_static_table(
            &stream,
            LaunchConfig::for_num_elems(1),
            static_out_dev.cu_deviceptr() as *mut f32,
        )
    }
    .expect("Static table kernel launch failed");
    let static_result = static_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy static result")[0];
    let static_expected = 12.0f32;
    println!("Static table: result = {static_result}");
    if (static_result - static_expected).abs() > f32::EPSILON {
        eprintln!("FAILED: expected {static_expected}, got {static_result}");
        std::process::exit(1);
    }

    println!("\nSUCCESS: device globals and non-zero static table behaved correctly.");
}
