/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `core::mem::swap` / `mem::replace` on the device (regression).
//!
//! `mem::swap`/`mem::replace` lower to the compiler intrinsic
//! `core::intrinsics::typed_swap_nonoverlapping`, which has no MIR body. Before
//! the fix this bailed with
//! `rustc intrinsic core::intrinsics::typed_swap_nonoverlapping is not yet
//! supported on the device`. The fix lowers it as a temp-free load/load/
//! store/store crossover.
//!
//! Each lane loads `a[i]` and `b[i]` into two locals, `mem::swap`s them, then
//! `mem::replace`s a sentinel into one. The result is encoded so the host can
//! verify both operations:
//!   out[i] = x*1000 + old = b[i]*1000 + a[i]
//!
//! Run: cargo oxide run mem_swap

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn swap_kernel(a: &[i32], b: &[i32], mut out: DisjointSlice<i32>) {
        let t = thread::index_1d();
        let i = t.get();
        let mut x = a[i];
        let mut y = b[i];
        // mem::swap -> typed_swap_nonoverlapping(&mut x, &mut y)
        core::mem::swap(&mut x, &mut y); // now x = b[i], y = a[i]
        // mem::replace also routes through the swap intrinsic:
        let old = core::mem::replace(&mut y, 7); // old = a[i], y = 7
        if let Some(slot) = out.get_mut(t) {
            // encode both: x (= b[i]) and old (= a[i])
            *slot = x * 1000 + old;
        }
    }
}

const N: usize = 64;

fn main() {
    println!("=== core::mem::swap / mem::replace on device ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/mem_swap.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX (device codegen failed?)");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");
    let stream = ctx.default_stream();

    let a_init: Vec<i32> = (0..N as i32).collect(); // a[i] = i
    let b_init: Vec<i32> = (0..N as i32).map(|i| 100 + i).collect(); // b[i] = 100 + i
    let d_a = DeviceBuffer::from_host(&stream, &a_init).unwrap();
    let d_b = DeviceBuffer::from_host(&stream, &b_init).unwrap();
    let mut d_out = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (N as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.swap_kernel(stream.as_ref(), config, &d_a, &d_b, &mut d_out) }
        .expect("Kernel launch failed");

    let out = d_out.to_host_vec(&stream).unwrap();
    let mut ok = true;
    for (i, got) in out.iter().enumerate() {
        let want = (100 + i as i32) * 1000 + i as i32; // b[i]*1000 + a[i]
        if *got != want {
            println!("FAIL: lane {i}: out={got} (want {want})");
            ok = false;
            break;
        }
    }
    if ok {
        println!("SUCCESS: all {N} lanes swapped+replaced correctly (out = b*1000 + a)");
    } else {
        std::process::exit(1);
    }
}
