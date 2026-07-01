/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, ptx_asm};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn inline_ptx_kernel(mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            let i = idx.get() as u32;
            let rust_before = i.wrapping_add(3);
            let doubled: u32;
            let lane: u32;

            unsafe {
                ptx_asm!(
                    "add.u32 %0, %1, %1;",
                    out("=r") doubled,
                    in("r") rust_before,
                    options(register_only),
                );
                ptx_asm!("mov.u32 %0, %%laneid;", out("=r") lane);
                ptx_asm!("membar.gl;", clobber("memory"));
            }

            let rust_after = doubled.wrapping_sub(3).wrapping_add(lane);
            *slot = rust_after;
        }
    }
}

fn main() {
    println!("=== Inline PTX Example ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 128;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe {
        module.inline_ptx_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
    }
    .expect("Kernel launch failed");

    let out = out_dev.to_host_vec(&stream).unwrap();
    for (i, got) in out.iter().copied().enumerate() {
        let expected = (i as u32 * 2) + 3 + (i as u32 % 32);
        if got != expected {
            eprintln!("Mismatch at {i}: expected {expected}, got {got}");
            std::process::exit(1);
        }
    }

    println!("SUCCESS: inline PTX results are correct");
}
