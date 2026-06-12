/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `core::ptr::copy_nonoverlapping` lowering.
//!
//! `copy_nonoverlapping` lowers to a MIR `StatementKind::Intrinsic(
//! NonDivergingIntrinsic::CopyNonOverlapping(_))`. The mir-importer's
//! statement translator used to reject this with a clear diagnostic because the
//! previous catch-all silently dropped the statement, producing PTX where the
//! memcpy was completely absent.
//!
//! Usage:
//!   cargo oxide run copy_nonoverlapping
//!
//! Expected: build succeeds and the runtime copy check passes.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, DisjointSlice};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn copy_nonoverlapping_kernel(input: &[u32], mut out: DisjointSlice<u32>) {
        if let Some((slot, idx)) = out.get_mut_indexed() {
            unsafe {
                let src = input.as_ptr().add(idx.get());
                let dst = slot as *mut u32;
                core::ptr::copy_nonoverlapping(src, dst, 1);
            }
        }
    }
}

fn main() {
    println!("=== copy_nonoverlapping ===");

    const N: usize = 128;
    let input_host: Vec<u32> = (0..N as u32)
        .map(|i| i.wrapping_mul(17) ^ 0x55aa_1234)
        .collect();

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let input_dev = DeviceBuffer::from_host(&stream, &input_host).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .copy_nonoverlapping_kernel(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &input_dev,
            &mut out_dev,
        )
        .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();
    assert_eq!(out_host, input_host);
    println!("PASS: copy_nonoverlapping copied {N} u32 values on the device");
}
