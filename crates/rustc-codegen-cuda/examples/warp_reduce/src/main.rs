/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(clippy::needless_range_loop)]

//! Unified Warp Reduction Example
//!
//! Demonstrates warp-level primitives: shuffle_xor, shuffle_down, shuffle (broadcast).
//! Uses butterfly reduction pattern for efficient parallel sum.
//!
//! Build and run with:
//!   cargo oxide run warp_reduce

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Warp reduction using shuffle_xor (butterfly pattern).
    /// All lanes end up with the complete sum.
    #[kernel]
    pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        // Load value (or 0 if out of bounds)
        let mut val = if gid.in_bounds(out.len() * 32) {
            data[gid.get()]
        } else {
            0.0
        };

        // Butterfly reduction using shuffle_xor
        val = val + warp::shuffle_xor_f32(val, 16);
        val = val + warp::shuffle_xor_f32(val, 8);
        val = val + warp::shuffle_xor_f32(val, 4);
        val = val + warp::shuffle_xor_f32(val, 2);
        val = val + warp::shuffle_xor_f32(val, 1);

        // Lane 0 writes the result
        if lane == 0 {
            let warp_idx = gid.get() / 32;
            unsafe {
                *out.get_unchecked_mut(warp_idx) = val;
            }
        }
    }

    /// Warp reduction using shuffle_down (sequential pattern).
    /// Only lane 0 has the complete sum.
    #[kernel]
    pub fn warp_reduce_sum_down(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        let mut val = if gid.in_bounds(out.len() * 32) {
            data[gid.get()]
        } else {
            0.0
        };

        // Sequential reduction using shuffle_down
        val = val + warp::shuffle_down_f32(val, 16);
        val = val + warp::shuffle_down_f32(val, 8);
        val = val + warp::shuffle_down_f32(val, 4);
        val = val + warp::shuffle_down_f32(val, 2);
        val = val + warp::shuffle_down_f32(val, 1);

        // Only lane 0 has the complete sum
        if lane == 0 {
            let warp_idx = gid.get() / 32;
            unsafe {
                *out.get_unchecked_mut(warp_idx) = val;
            }
        }
    }

    /// Broadcast kernel - shuffle lane 0's value to all lanes.
    #[kernel]
    pub fn warp_broadcast(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();

        let my_val = if gid.in_bounds(out.len()) {
            data[gid.get()]
        } else {
            0.0
        };

        // Broadcast lane 0's value to all lanes
        let broadcast_val = warp::shuffle_f32(my_val, 0);

        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = broadcast_val;
        }
    }

    /// Test lane_id() intrinsic - each thread writes its lane ID.
    #[kernel]
    pub fn test_lane_id(mut out: DisjointSlice<u32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();

        if gid.in_bounds(out.len()) {
            // Each thread writes its lane ID
            unsafe {
                *out.get_unchecked_mut(gid.get()) = lane;
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Warp Reduction Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 256;
    const WARPS: usize = N / 32;

    // Initialize input: each warp has values 0-31, so sum = 496
    let data_host: Vec<f32> = (0..N).map(|i| (i % 32) as f32).collect();

    println!("Input data: {} elements, {} warps", N, WARPS);
    println!("  First warp values: {:?}", &data_host[0..8]);
    println!("  Expected warp sum: {}", (0..32).sum::<i32>());

    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();

    let module = ctx
        .load_module_from_file("warp_reduce.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (WARPS as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: Butterfly reduction =====
    println!("\n--- Test 1: Butterfly Reduction (shuffle_xor) ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_sum((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    let expected_sum = (0..32).sum::<i32>() as f32;
    let all_correct = out_result.iter().all(|&x| (x - expected_sum).abs() < 1e-5);

    if all_correct {
        println!(
            "✓ All {} warp sums correct (each = {})",
            WARPS, expected_sum
        );
    } else {
        println!("✗ Some warp sums incorrect!");
        std::process::exit(1);
    }

    // ===== Test 2: Sequential reduction =====
    println!("\n--- Test 2: Sequential Reduction (shuffle_down) ---");
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, WARPS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.warp_reduce_sum_down((stream).as_ref(), cfg, &data_dev, &mut out_dev) }
        .expect("Kernel launch failed");

    let out_result = out_dev.to_host_vec(&stream).unwrap();
    println!("Warp sums: {:?}", out_result);

    let all_correct = out_result.iter().all(|&x| (x - expected_sum).abs() < 1e-5);
    if all_correct {
        println!(
            "✓ All {} warp sums correct (each = {})",
            WARPS, expected_sum
        );
    } else {
        println!("✗ Some warp sums incorrect!");
        std::process::exit(1);
    }

    // ===== Test 3: Broadcast =====
    println!("\n--- Test 3: Broadcast (shuffle to lane 0) ---");
    let broadcast_input: Vec<f32> = (0..N).map(|i| (i * 10) as f32).collect();
    let broadcast_dev = DeviceBuffer::from_host(&stream, &broadcast_input).unwrap();
    let mut broadcast_out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.warp_broadcast(
            (stream).as_ref(),
            cfg,
            &broadcast_dev,
            &mut broadcast_out_dev,
        )
    }
    .expect("Kernel launch failed");

    let broadcast_result = broadcast_out_dev.to_host_vec(&stream).unwrap();

    println!("Broadcast results (first 8 of each warp):");
    for warp in 0..WARPS.min(4) {
        let start = warp * 32;
        println!("  Warp {}: {:?}", warp, &broadcast_result[start..start + 8]);
    }

    // Verify each warp has all same values
    let mut broadcast_correct = true;
    for warp in 0..WARPS {
        let start = warp * 32;
        let expected = broadcast_input[start]; // Lane 0's value
        for lane in 0..32 {
            if (broadcast_result[start + lane] - expected).abs() > 1e-5 {
                println!(
                    "Mismatch: warp {} lane {} expected {} got {}",
                    warp,
                    lane,
                    expected,
                    broadcast_result[start + lane]
                );
                broadcast_correct = false;
            }
        }
    }

    if broadcast_correct {
        println!("✓ Broadcast correct: all lanes have lane 0's value");
    } else {
        println!("✗ Broadcast failed!");
        std::process::exit(1);
    }

    // ===== Test 4: Lane ID =====
    println!("\n--- Test 4: Lane ID ---");
    let mut lane_out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.test_lane_id((stream).as_ref(), cfg, &mut lane_out_dev) }
        .expect("Kernel launch failed");

    let lane_result = lane_out_dev.to_host_vec(&stream).unwrap();

    println!("Lane IDs (first 8 of each of first 2 warps):");
    for warp in 0..2.min(WARPS) {
        let start = warp * 32;
        println!("  Warp {}: {:?}", warp, &lane_result[start..start + 8]);
    }

    // Verify lane IDs are 0-31 repeating for each warp
    let mut lane_correct = true;
    for i in 0..N {
        let expected_lane = (i % 32) as u32;
        if lane_result[i] != expected_lane {
            println!(
                "Mismatch at {}: expected lane {} got {}",
                i, expected_lane, lane_result[i]
            );
            lane_correct = false;
        }
    }

    if lane_correct {
        println!("✓ Lane IDs correct: 0-31 pattern for each warp");
    } else {
        println!("✗ Lane ID test failed!");
        std::process::exit(1);
    }

    println!("\n✓ SUCCESS: All warp tests passed!");
}
