/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `core::ptr::copy` (the overlap-safe move).
//!
//! Unlike `copy_nonoverlapping` (which reaches MIR as a `CopyNonOverlapping`
//! statement and already lowers to `llvm.memcpy`), `core::ptr::copy` bottoms
//! out in the `core::intrinsics::copy` intrinsic, which the device backend did
//! not lower — codegen failed with "rustc intrinsic `copy` is not yet supported
//! on the device". libcore reaches it from `ptr::swap`, `slice` rotates, and
//! similar routines. The fix lowers it to the overlap-safe `llvm.memmove`.
//!
//! Usage:
//!   cargo oxide run ptr_copy

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Thread 0 loads `input` into `out`, then shifts `out[0..n-1]` up by one
    /// with `ptr::copy` (dst = src + 1, a forward-overlapping move that a plain
    /// memcpy would corrupt). Result: `out[0] == input[0]`, `out[k] == input[k-1]`.
    #[kernel]
    pub fn shift_right_one(input: &[i32], mut out: DisjointSlice<i32>, n: usize) {
        if thread::index_1d().get() == 0 {
            unsafe {
                let p = out.as_mut_ptr();
                core::ptr::copy_nonoverlapping(input.as_ptr(), p, n);
                core::ptr::copy(p, p.add(1), n - 1);
            }
        }
    }
}

fn main() {
    println!("=== ptr_copy ===");
    const N: usize = 96;
    let input: Vec<i32> = (0..N as i32).map(|i| i * 3 - 7).collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    let din = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.shift_right_one(&stream, cfg, &din, &mut out, N) }
        .expect("shift_right_one launch");
    let got = out.to_host_vec(&stream).unwrap();

    let mut want = input.clone();
    for k in (1..N).rev() {
        want[k] = want[k - 1];
    }
    assert_eq!(got, want, "shift_right_one (overlapping ptr::copy)");
    println!("PASS: ptr_copy (overlapping forward move via memmove)");
}
