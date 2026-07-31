/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only BF16 WGMMA MMA example.
//!
//! This example exercises the deferred 32-register accumulator adapter. The
//! compiler recognizes the linear fence/MMA/commit/wait sequence and emits one
//! convergent inline-PTX block that stores the accumulator only after
//! `wgmma.wait_group.sync.aligned 0`.
//!
//! Usage:
//!   cargo oxide build wgmma_mma_bf16 --arch sm_90a
//!
//! The descriptors are intentionally zero because the example validates
//! compilation and PTX generation only. Do not launch this kernel.

use cuda_device::wgmma::{
    wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16, wgmma_wait_group,
};
use cuda_device::{DisjointSlice, kernel, thread};

/// # Safety
///
/// This kernel is compile-only. Its zero descriptors are not valid WGMMA
/// shared-memory descriptors and must not be executed.
#[kernel]
pub unsafe fn wgmma_mma_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];

    unsafe {
        wgmma_fence();
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
        wgmma_commit_group();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

fn main() {
    println!("SUCCESS: BF16 WGMMA deferred accumulator lowering compiled.");
}
