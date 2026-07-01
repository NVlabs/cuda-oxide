/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Vector Addition Example
//!
//! THIS IS THE GOAL: Single file, single compilation, no cfg splits.
//!
//! Build and run with:
//!   cargo oxide run vecadd
//!
//! What happens:
//! 1. rustc parses this file, generates MIR for everything
//! 2. rustc-codegen-cuda intercepts codegen:
//!    - Finds `cuda_oxide_kernel_<hash>_vecadd` (from #[kernel])
//!    - Routes it to mir-importer → PTX
//!    - Routes `main` and other host code to standard LLVM
//! 3. Final binary has both host code and embedded PTX

// No #![cfg_attr(cuda_device, no_std)] - this compiles as ONE unit!

use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_program, kernel, thread};
use cuda_host::ProgramLowering;

// =============================================================================
// KERNEL - This gets compiled to PTX by rustc-codegen-cuda
// =============================================================================

/// Vector addition kernel: c[i] = a[i] + b[i]
///
/// This function exists in BOTH host MIR and device PTX:
/// - Host: The function body is never called, but types are checked
/// - Device: Compiled to PTX via mir-importer pipeline
#[cuda_program]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    #[kernel]
    pub fn scale(input: &[f32], mut out: DisjointSlice<f32>, factor: f32) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = input[idx_raw] * factor;
        }
    }

    #[program]
    pub fn forward(
        a: &DeviceBuffer<f32>,
        b: &DeviceBuffer<f32>,
        tmp: &mut DeviceBuffer<f32>,
        c: &mut DeviceBuffer<f32>,
        n: u32,
    ) {
        vecadd(a, b, tmp).grid_len(n);
        scale(tmp, c, 1.0f32).grid_len(n);
    }
}

// =============================================================================
// HOST CODE - This gets compiled to native x86_64 by LLVM
// =============================================================================

fn main() {
    println!("=== Unified Compilation Vector Addition ===\n");

    // Initialize CUDA
    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    // Test data
    const N: usize = 1024;
    let a_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let b_host: Vec<f32> = (0..N).map(|i| (i * 2) as f32).collect();

    println!("Input vectors (first 5 elements):");
    println!("  a = {:?}", &a_host[0..5]);
    println!("  b = {:?}", &b_host[0..5]);

    // Allocate device memory
    let a_dev = DeviceBuffer::from_host(&stream, &a_host).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_host).unwrap();
    let mut tmp_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // Load the embedded PTX bundle and launch through the typed module API.
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let graph = kernels::forward_graph(&a_dev, &b_dev, &mut tmp_dev, &mut c_dev, N as u32);
    assert_eq!(graph.dependencies().len(), 1);
    assert_eq!(graph.dependencies()[0].resource, "tmp");
    let bound = graph
        .bind(&module, ProgramLowering::SequentialLaunches)
        .expect("Failed to bind CUDA program graph");
    bound
        .launch(&stream)
        .expect("Kernel launch failed");

    // Get results
    let c_host = c_dev.to_host_vec(&stream).unwrap();

    println!("\nOutput vector (first 5 elements):");
    println!("  c = {:?}", &c_host[0..5]);

    // Verify
    let mut errors = 0;
    for i in 0..N {
        let expected = a_host[i] + b_host[i];
        if (c_host[i] - expected).abs() > 1e-5 {
            if errors < 5 {
                eprintln!(
                    "  Error at [{}]: expected {}, got {}",
                    i, expected, c_host[i]
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("\n✓ SUCCESS: All {} elements correct!", N);
    } else {
        println!("\n✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
