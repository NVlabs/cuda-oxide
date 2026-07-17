/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Unified Barrier Test Example
//!
//! Demonstrates mbarrier (async barrier) intrinsics:
//! - mbarrier_init: Initialize barrier
//! - mbarrier_arrive: Arrive at barrier, get token
//! - mbarrier_wait: Wait for the phase using generated mbarrier_test_wait
//! - mbarrier_inval: Invalidate barrier
//!
//! Build and run with:
//!   cargo oxide run barrier

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::barrier::{
    Barrier, mbarrier_arrive, mbarrier_init, mbarrier_inval, mbarrier_wait,
};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Simple barrier test: all threads sync via mbarrier.
    #[kernel]
    pub fn barrier_sync_test(mut out: DisjointSlice<u32>) {
        static mut BAR: Barrier = Barrier::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // Thread 0 initializes the barrier
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
            }
        }

        // Ensure all threads see the initialized barrier
        thread::sync_threads();

        // Each thread arrives at the barrier
        let token = unsafe { mbarrier_arrive(&raw const BAR) };

        // Wait until the generated test-wait intrinsic reports completion.
        unsafe { mbarrier_wait(&raw const BAR, token) }

        // All threads reached here - barrier complete
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = 1u32;
        }

        // Ensure every thread is done before invalidation.
        thread::sync_threads();

        // Thread 0 invalidates the barrier
        if tid == 0 {
            unsafe {
                mbarrier_inval(&raw mut BAR);
            }
        }
    }

    /// Test with shared memory data and barrier synchronization.
    #[kernel]
    pub fn barrier_shared_data_test(mut out: DisjointSlice<u32>) {
        static mut BAR: Barrier = Barrier::UNINIT;
        static mut DATA: SharedArray<u32, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x();
        let block_size = thread::blockDim_x();
        let gid = thread::index_1d();

        // Thread 0 initializes barrier
        if tid == 0 {
            unsafe {
                mbarrier_init(&raw mut BAR, block_size);
            }
        }
        thread::sync_threads();

        // Write to shared memory
        unsafe {
            DATA[tid as usize] = tid;
        }

        // Arrive and wait for all threads
        let token = unsafe { mbarrier_arrive(&raw const BAR) };
        unsafe { mbarrier_wait(&raw const BAR, token) }

        // Read neighbor's data (with wraparound)
        let neighbor_idx = ((tid + 1) % block_size) as usize;
        let neighbor_val = unsafe { DATA[neighbor_idx] };

        // Output should be: [1, 2, 3, ..., 255, 0] for block_size=256
        if let Some(out_elem) = out.get_mut(gid) {
            *out_elem = neighbor_val;
        }

        // Ensure every thread is done before invalidation.
        thread::sync_threads();

        // Cleanup
        if tid == 0 {
            unsafe {
                mbarrier_inval(&raw mut BAR);
            }
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== Unified Barrier Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    let module = ctx
        .load_module_from_file("barrier.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    const N: usize = 256;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Test 1: barrier_sync_test
    println!("--- Test 1: barrier_sync_test ---");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.barrier_sync_test((stream).as_ref(), cfg, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        let all_ones = result.iter().all(|&x| x == 1);
        if all_ones {
            println!("✓ All {} threads completed barrier sync", N);
        } else {
            let failures: Vec<_> = result
                .iter()
                .enumerate()
                .filter(|&(_, &x)| x != 1)
                .collect();
            println!(
                "✗ Barrier sync failed! {} failures: {:?}",
                failures.len(),
                &failures[..failures.len().min(10)]
            );
            std::process::exit(1);
        }
    }

    // Test 2: barrier_shared_data_test
    println!("\n--- Test 2: barrier_shared_data_test ---");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

        // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
        unsafe { module.barrier_shared_data_test((stream).as_ref(), cfg, &mut out_dev) }
            .expect("Kernel launch failed");

        stream.synchronize().unwrap();
        let result = out_dev.to_host_vec(&stream).unwrap();

        // Verify pattern: [1, 2, 3, ..., 255, 0]
        let correct = result
            .iter()
            .enumerate()
            .all(|(i, &x)| x == ((i + 1) % N) as u32);

        if correct {
            println!("✓ Shared memory + barrier pattern correct");
            println!("  First 8 values: {:?}", &result[..8]);
            println!("  Last 3 values: {:?}", &result[N - 3..]);
        } else {
            println!("✗ Pattern mismatch!");
            let mismatches: Vec<_> = result
                .iter()
                .enumerate()
                .filter(|&(ref i, &x)| x != ((*i + 1) % N) as u32)
                .take(10)
                .collect();
            println!("  First mismatches: {:?}", mismatches);
            std::process::exit(1);
        }
    }

    println!("\n✓ SUCCESS: All barrier tests passed!");
}
