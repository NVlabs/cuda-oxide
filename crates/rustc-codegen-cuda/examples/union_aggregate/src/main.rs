/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for MIR union aggregate construction.
//!
//! `Bits { word: value }` is a union aggregate in MIR. The importer must lower
//! the active field and fill inactive storage slots with undef instead of
//! rejecting union aggregate construction.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[allow(dead_code)]
union Bits {
    word: u32,
    bytes: [u8; 4],
}

#[inline(never)]
fn make_bits(word: u32) -> Bits {
    Bits { word }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn union_aggregate(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get() as u32;
        if let Some(slot) = out.get_mut(idx) {
            let bits = make_bits(0x5a00_0000 | raw);
            unsafe {
                *slot = bits.word;
            }
        }
    }
}

fn main() {
    const N: usize = 32;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .union_aggregate(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out_dev,
        )
        .expect("kernel launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, got) in out.iter().enumerate() {
        let expected = 0x5a00_0000 | i as u32;
        if *got != expected {
            eprintln!("FAIL lane {i}: got {got:#x}, expected {expected:#x}");
            std::process::exit(1);
        }
    }

    println!("union_aggregate: PASS");
}
