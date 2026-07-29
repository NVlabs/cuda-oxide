/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! ConstantIndex `from_end` smoke test.
//!
//! Forces rustc to emit `ProjectionElem::ConstantIndex { from_end: true }`
//! via rest patterns on fixed-length arrays (`let [.., last] = arr` and
//! `let [.., ref mut last] = arr`). Before support landed, these rejected
//! with "ConstantIndex with from_end=true not yet supported".
//!
//! Run: cargo oxide run constant_index_from_end

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Read the last element via `let [.., last] = arr`.
    #[kernel]
    pub fn read_last(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let arr: [u32; 4] = [10, 20, 30, 40];
            let [.., last] = arr;
            *out_elem = last;
        }
    }

    /// Write through a from-end mutable binding, then read last + first.
    #[kernel]
    pub fn write_last(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut arr: [u32; 4] = [1, 2, 3, 4];
            let [.., ref mut last] = arr;
            *last = val;
            *out_elem = arr[0] + arr[3];
        }
    }
}

fn main() {
    const N: usize = 64;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    // --- read_last ---
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.read_last(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
    }
    .expect("read_last launch");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let mut errors = 0usize;
    for (i, &val) in out.iter().enumerate() {
        if val != 40 {
            if errors < 5 {
                eprintln!("  FAIL read_last[{}]: got {} want 40", i, val);
            }
            errors += 1;
        }
    }

    // --- write_last ---
    let val: u32 = 99;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    unsafe {
        module.write_last(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            val,
            &mut out_dev,
        )
    }
    .expect("write_last launch");
    let out = out_dev.to_host_vec(&stream).unwrap();
    let expected = 1 + val; // arr[0] + arr[3]
    for (i, &got) in out.iter().enumerate() {
        if got != expected {
            if errors < 5 {
                eprintln!("  FAIL write_last[{}]: got {} want {}", i, got, expected);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("PASS: constant_index_from_end (read + write)");
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
