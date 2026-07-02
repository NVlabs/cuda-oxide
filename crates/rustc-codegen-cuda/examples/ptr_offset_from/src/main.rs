// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Regression coverage for raw pointer distance intrinsics.
//!
//! The stable raw pointer methods lower through rustc intrinsics in MIR:
//! `offset_from_unsigned` uses `ptr_offset_from_unsigned`, while
//! `offset_from` and the byte-oriented wrappers use `ptr_offset_from`.
//! cuda-oxide must compute the byte address difference and divide by the
//! pointee size, preserving signed results for negative distances.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn pointer_distances(values: &[u32], lo: usize, hi: usize, mut out: DisjointSlice<i64>) {
        if thread::index_1d().get() == 0 {
            let base = values.as_ptr();
            unsafe {
                let lo_ptr = base.add(lo);
                let hi_ptr = base.add(hi);

                *out.get_unchecked_mut(0) = hi_ptr.offset_from_unsigned(lo_ptr) as i64;
                *out.get_unchecked_mut(1) = lo_ptr.offset_from(hi_ptr) as i64;
                *out.get_unchecked_mut(2) = hi_ptr.byte_offset_from(lo_ptr) as i64;
                *out.get_unchecked_mut(3) = hi_ptr.byte_offset_from_unsigned(lo_ptr) as i64;
            }
        }
    }
}

const SLOTS: usize = 4;

fn main() {
    let ctx = CudaContext::new(0).expect("ctx");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load");

    let values: Vec<u32> = (0..16).map(|v| v * 11).collect();
    let lo = 3usize;
    let hi = 9usize;

    let values_dev = DeviceBuffer::from_host(&stream, &values).unwrap();
    let mut out_dev = DeviceBuffer::<i64>::zeroed(&stream, SLOTS).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.pointer_distances(
            &stream,
            LaunchConfig::for_num_elems(1),
            &values_dev,
            lo,
            hi,
            &mut out_dev,
        )
    }
    .expect("launch pointer_distances");

    let got = out_dev.to_host_vec(&stream).unwrap();
    let element_distance = (hi - lo) as i64;
    let byte_distance = ((hi - lo) * core::mem::size_of::<u32>()) as i64;
    let expected = [
        element_distance,
        -element_distance,
        byte_distance,
        byte_distance,
    ];

    let mut errors = 0;
    for (i, (&actual, &want)) in got.iter().zip(expected.iter()).enumerate() {
        if actual != want {
            eprintln!("slot {i}: expected {want}, got {actual}");
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: pointer distances are correct");
    } else {
        println!("FAILURE: {errors} pointer distance mismatches");
        std::process::exit(1);
    }
}
