/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed-API host-closure smoke test.
//!
//! Drives the typed `#[cuda_module]` launch path with a generic kernel that
//! takes a closure. This exercises the kernel-boundary ABI fix that lets the
//! typed path push the closure as a single byval `.param` and have the
//! backend emit a single matching `.param` declaration. Before the fix this
//! example only worked through the call-site `cuda_launch!` macro because
//! that path pushed each capture individually; the typed API kept the
//! closure intact and mis-matched the backend's flattened ABI.
//!
//! Build and run with:
//!   cargo oxide run host_closure
//!
//! ## What it covers
//!
//! 1. Generic kernel with `Fn` trait bound: `fn map<T, F: Fn(T) -> T + Copy>(...)`
//! 2. Closure with 0, 1, 2, 3, 4 captures.
//! 3. Type inference of `F` at the call site (`module.map::<f32, _>(...)`).
//! 4. The closure is pushed via `push_kernel_scalar` and read on the device as
//!    a single byval struct — no per-capture flattening on either side.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

// =============================================================================
// CLOSURE-ACCEPTING GENERIC KERNEL
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Generic map kernel — applies a function to each element.
    ///
    /// `F` is bound to the closure's anonymous type at the call site via
    /// the typed API's turbofish placeholder (`module.map::<f32, _>(...)`).
    /// The closure value itself is pushed as one byval kernel argument; the
    /// device reads it back as `F` and calls `f(input[idx])` on each thread.
    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[idx_raw]);
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Typed Closure Kernel Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let input_data: Vec<f32> = (0..N).map(|i| i as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_data)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let module = kernels::load(&ctx)
        .map_err(|err| format!("failed to load host_closure kernel module: {err}"))?;

    let cfg = LaunchConfig::for_num_elems(N as u32);
    let mut failed = false;

    // =========================================================================
    // TEST 1: Closure with single capture
    // =========================================================================
    println!("Test 1: Single capture (scale by factor)");
    {
        let factor = 2.5f32;
        println!("  factor = {}", factor);
        println!("  N = {}", N);

        module.map::<f32, _>(
            &stream,
            cfg,
            move |x: f32| x * factor,
            &input_dev,
            &mut output_dev,
        )?;

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - input_data[i] * factor).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 2: Closure with multiple captures
    // =========================================================================
    println!("Test 2: Multiple captures (affine transform)");
    {
        let scale = 2.0f32;
        let offset = 10.0f32;
        println!("  scale = {}, offset = {}", scale, offset);

        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        module.map::<f32, _>(
            &stream,
            cfg,
            move |x: f32| x * scale + offset,
            &input_dev,
            &mut output_dev,
        )?;

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - (input_data[i] * scale + offset)).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 3: Zero-capture closure (inline constant)
    // =========================================================================
    println!("Test 3: Zero captures (double each element)");
    {
        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        module.map::<f32, _>(&stream, cfg, |x: f32| x * 2.0, &input_dev, &mut output_dev)?;

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| (output_host[i] - input_data[i] * 2.0).abs() > 1e-5)
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
        }
    }

    // =========================================================================
    // TEST 4: Closure with 3 captures (polynomial transform)
    // =========================================================================
    println!("Test 4: Three captures (polynomial: a*x^2 + b*x + c)");
    {
        let a = 0.5f32;
        let b = 2.0f32;
        let c = 1.0f32;
        println!("  a = {}, b = {}, c = {}", a, b, c);

        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        module.map::<f32, _>(
            &stream,
            cfg,
            move |x: f32| a * x * x + b * x + c,
            &input_dev,
            &mut output_dev,
        )?;

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| {
                let x = input_data[i];
                let expected = a * x * x + b * x + c;
                (output_host[i] - expected).abs() > 1e-3
            })
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
            for i in 0..N.min(5) {
                let x = input_data[i];
                let expected = a * x * x + b * x + c;
                println!("    [{i}]: got {}, expected {}", output_host[i], expected);
            }
        }
    }

    // =========================================================================
    // TEST 5: Closure with 4 captures (to ensure arbitrary count works)
    // =========================================================================
    println!("Test 5: Four captures (weighted sum: w1*x + w2 + w3*w4)");
    {
        let w1 = 3.0f32;
        let w2 = 5.0f32;
        let w3 = 2.0f32;
        let w4 = 7.0f32;
        println!("  w1 = {}, w2 = {}, w3 = {}, w4 = {}", w1, w2, w3, w4);

        output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

        module.map::<f32, _>(
            &stream,
            cfg,
            move |x: f32| w1 * x + w2 + w3 * w4,
            &input_dev,
            &mut output_dev,
        )?;

        let output_host = output_dev.to_host_vec(&stream)?;
        let errors = (0..N)
            .filter(|&i| {
                let x = input_data[i];
                let expected = w1 * x + w2 + w3 * w4;
                (output_host[i] - expected).abs() > 1e-3
            })
            .count();

        if errors == 0 {
            println!("  ✓ SUCCESS: All {} elements correct!\n", N);
        } else {
            println!("  ✗ FAILED: {} errors\n", errors);
            failed = true;
            for i in 0..N.min(5) {
                let x = input_data[i];
                let expected = w1 * x + w2 + w3 * w4;
                println!("    [{i}]: got {}, expected {}", output_host[i], expected);
            }
        }
    }

    println!("=== All Tests Complete ===");
    if failed {
        Err("one or more host_closure tests failed".into())
    } else {
        Ok(())
    }
}
