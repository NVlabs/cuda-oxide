/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end test for `cp.async` 4-byte and 8-byte copy intrinsics.
//!
//! Demonstrates asynchronous global-to-shared memory copies using
//! `cp_async_ca_4` (4 bytes) and `cp_async_ca_8` (8 bytes), with
//! `cp.async.commit_group` / `cp.async.wait_all` issued via inline PTX.
//!
//! Requires **sm_80+** (Ampere or later).
//!
//! Build and run with:
//!   cargo oxide run cp_async_small

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::async_copy::{cp_async_ca_4, cp_async_ca_8};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, ptx_asm, thread};

// =============================================================================
// KERNELS
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread copies one `u32` from global to shared memory via
    /// `cp.async.ca.shared.global [...], [...], 4`, then writes it out.
    #[kernel]
    pub fn test_cp_async_4(input: &[u32], mut out: DisjointSlice<u32>) {
        static mut SMEM: SharedArray<u32, 32> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d();

        // Obtain shared-memory and global-memory pointers.
        let dst_ptr = unsafe { SMEM.as_mut_ptr().add(tid) };
        let src_ptr = unsafe { input.as_ptr().add(gid.get()) };

        // Initiate the 4-byte async copy, commit, and wait.
        unsafe {
            cp_async_ca_4(dst_ptr, src_ptr);
            ptx_asm!("cp.async.commit_group;");
            ptx_asm!("cp.async.wait_all;");
        }

        // Ensure every thread's copy has landed before reading.
        thread::sync_threads();

        // Read back from shared memory and write to global output.
        let val = unsafe { SMEM[tid] };
        if let Some(slot) = out.get_mut(gid) {
            *slot = val;
        }
    }

    /// Each thread copies 8 bytes (two consecutive `u32`s) from global to
    /// shared memory via `cp.async.ca.shared.global [...], [...], 8`, then
    /// writes both values out.
    #[kernel]
    pub fn test_cp_async_8(input: &[u32], mut out: DisjointSlice<u32>) {
        // 64 elements: thread i owns elements [2*i] and [2*i+1].
        static mut SMEM: SharedArray<u32, 64> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let gid = thread::index_1d().get();

        let smem_idx = tid * 2;
        let gmem_idx = gid * 2;

        let dst_ptr = unsafe { SMEM.as_mut_ptr().add(smem_idx) };
        let src_ptr = unsafe { input.as_ptr().add(gmem_idx) };

        // Initiate the 8-byte async copy, commit, and wait.
        unsafe {
            cp_async_ca_8(dst_ptr, src_ptr);
            ptx_asm!("cp.async.commit_group;");
            ptx_asm!("cp.async.wait_all;");
        }

        thread::sync_threads();

        // Write both copied values to the output buffer.
        // Each thread writes to two unique slots [2*gid] and [2*gid+1].
        unsafe {
            let lo = SMEM[smem_idx];
            let hi = SMEM[smem_idx + 1];
            let base = gid * 2;
            *out.get_unchecked_mut(base) = lo;
            *out.get_unchecked_mut(base + 1) = hi;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() {
    println!("=== cp.async small-copy example (4-byte and 8-byte) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");

    // cp.async requires sm_80 (Ampere) or later.
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 8 {
        println!(
            "Skipping: cp.async requires sm_80+, device is sm_{}{} -- PASS (skipped)",
            major, minor
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    let cfg32 = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    };

    // ===== Test 1: cp.async.ca 4-byte copy =====
    println!("--- Test 1: cp.async.ca 4-byte copy ---");
    {
        let input: Vec<u32> = (100..132).collect(); // 32 distinct values
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 32).unwrap();

        module
            .test_cp_async_4(&stream, cfg32, &input_dev, &mut out_dev)
            .expect("test_cp_async_4 launch failed");

        let out = out_dev.to_host_vec(&stream).unwrap();
        for i in 0..32 {
            if out[i] != input[i] {
                eprintln!(
                    "FAIL at [{}]: expected {}, got {} (4-byte copy)",
                    i, input[i], out[i]
                );
                std::process::exit(1);
            }
        }
        println!("  PASS: 32 elements copied correctly via cp.async 4-byte");
    }

    // ===== Test 2: cp.async.ca 8-byte copy =====
    println!("--- Test 2: cp.async.ca 8-byte copy ---");
    {
        let input: Vec<u32> = (200..264).collect(); // 64 values (32 threads x 2)
        let input_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 64).unwrap();

        module
            .test_cp_async_8(&stream, cfg32, &input_dev, &mut out_dev)
            .expect("test_cp_async_8 launch failed");

        let out = out_dev.to_host_vec(&stream).unwrap();
        for i in 0..64 {
            if out[i] != input[i] {
                eprintln!(
                    "FAIL at [{}]: expected {}, got {} (8-byte copy)",
                    i, input[i], out[i]
                );
                std::process::exit(1);
            }
        }
        println!("  PASS: 64 elements copied correctly via cp.async 8-byte");
    }

    println!("\nPASS: cp.async.ca 4-byte and 8-byte copies correct");
}
