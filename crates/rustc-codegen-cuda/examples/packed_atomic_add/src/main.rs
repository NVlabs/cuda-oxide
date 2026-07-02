/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end packed f16x2 and bf16x2 global atomic-add example.
//!
//! The native bf16x2 instruction makes this combined example require sm_90.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::atomic::{atom_add_bf16x2, atom_add_f16x2};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;
use half::{bf16, f16};

const THREADS: u32 = 32;
const PACKED_F16_ONE: u32 = 0x3c00_3c00;
const PACKED_BF16_ONE: u32 = 0x3f80_3f80;
const COUNTERS: usize = 2;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn add_packed(base: *mut u32) {
        if thread::index_1d().get() >= THREADS as usize {
            return;
        }

        unsafe {
            let _ = atom_add_f16x2(base, PACKED_F16_ONE);
            let _ = atom_add_bf16x2(base.add(1), PACKED_BF16_ONE);
        }
    }
}

fn unpack_f16x2(value: u32) -> (f32, f32) {
    (
        f16::from_bits(value as u16).to_f32(),
        f16::from_bits((value >> 16) as u16).to_f32(),
    )
}

fn unpack_bf16x2(value: u32) -> (f32, f32) {
    (
        bf16::from_bits(value as u16).to_f32(),
        bf16::from_bits((value >> 16) as u16).to_f32(),
    )
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA init");
    let (major, minor) = ctx.compute_capability().expect("compute capability");
    if major < 9 {
        println!(
            "skipping: native bf16x2 atomic add requires sm_90+ (device is sm_{major}{minor})"
        );
        return;
    }

    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded PTX");
    let counters = DeviceBuffer::<u32>::zeroed(&stream, COUNTERS).expect("allocate counters");
    let counters_ptr = counters.cu_deviceptr() as *mut u32;
    module
        .add_packed(&stream, LaunchConfig::for_num_elems(THREADS), counters_ptr)
        .expect("launch add_packed");
    stream.synchronize().expect("synchronize");

    let result = counters.to_host_vec(&stream).expect("copy counters");
    let expected = THREADS as f32;
    assert_eq!(unpack_f16x2(result[0]), (expected, expected));
    assert_eq!(unpack_bf16x2(result[1]), (expected, expected));
    println!("PASS: packed lane sums reached {expected} on sm_{major}{minor}");
}
