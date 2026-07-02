/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for assigning through nested runtime array indexes.
//!
//! MIR represents `local[i][j] = value` as a two-level `Index, Index`
//! projection. The statement translator must lower that chained projection to
//! an address and store through it instead of rejecting the assignment before
//! the generic projection walker can handle it.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn nested_index_assignment_kernel(i: usize, j: usize, mut out: DisjointSlice<u32>) {
        let mut values = [[0u32; 4]; 4];
        values[i][j] = 0x5a00_0000 | ((i as u32) << 8) | (j as u32);

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[i][j];
        }
    }

    // Non-square nested array: 5 rows of 3 columns. A square array cannot
    // catch a row-stride bug (row count == column count makes i and j
    // interchangeable), so this case proves the outer index uses the inner
    // array's real stride.
    #[kernel]
    pub fn nested_index_assignment_nonsquare_kernel(
        i: usize,
        j: usize,
        mut out: DisjointSlice<u32>,
    ) {
        let mut values = [[0u32; 3]; 5];
        values[i][j] = 0x5a00_0000 | ((i as u32) << 8) | (j as u32);

        if let Some((slot, _idx)) = out.get_mut_indexed() {
            *slot = values[i][j];
        }
    }
}

fn main() {
    println!("=== nested_index_assignment ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");

    // Square [[u32; 4]; 4], write [2][3].
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.nested_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            2usize,
            3usize,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");
    assert_eq!(out_dev.to_host_vec(&stream).unwrap(), vec![0x5a00_0203]);

    // Non-square [[u32; 3]; 5], write the last element [4][2].
    let mut out_ns = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.nested_index_assignment_nonsquare_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            4usize,
            2usize,
            &mut out_ns,
        )
    }
    .expect("Non-square kernel launch failed");
    assert_eq!(out_ns.to_host_vec(&stream).unwrap(), vec![0x5a00_0402]);

    println!("PASS: nested runtime indexes assigned and read back correctly (square + non-square)");
}
