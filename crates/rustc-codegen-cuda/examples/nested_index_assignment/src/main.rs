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
use cuda_device::{cuda_module, kernel, DisjointSlice};

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
}

fn main() {
    println!("=== nested_index_assignment ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 1).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .nested_index_assignment_kernel(
            &stream,
            LaunchConfig::for_num_elems(1),
            2usize,
            3usize,
            &mut out_dev,
        )
        .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    assert_eq!(out_host, vec![0x5a00_0203]);
    println!("PASS: nested runtime indexes assigned and read back correctly");
}
