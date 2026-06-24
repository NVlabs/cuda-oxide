/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for lowering function-item receivers in rust-call paths.
//!
//! Passing a function item to a generic `FnOnce` helper makes MIR call
//! `<fn item as FnOnce>::call_once`. The importer must resolve the receiver
//! back to the concrete function body instead of emitting a dangling trait-shim
//! callee symbol.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::cuda_module;

#[inline(never)]
#[device]
fn plus_seven(x: u32) -> u32 {
    x + 7
}

#[device]
fn apply_once<F: FnOnce(u32) -> u32>(f: F, x: u32) -> u32 {
    f(x)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn function_item_call(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let raw = idx.get() as u32;
        if let Some(slot) = out.get_mut(idx) {
            *slot = apply_once(plus_seven, raw);
        }
    }
}

fn main() {
    const N: usize = 16;

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .function_item_call(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out_dev,
        )
        .expect("kernel launch");

    let out = out_dev.to_host_vec(&stream).unwrap();
    let expected: Vec<u32> = (0..N).map(|i| i as u32 + 7).collect();
    if out != expected {
        eprintln!("FAIL: got {out:?}, expected {expected:?}");
        std::process::exit(1);
    }

    println!("function_item_call: PASS");
}
