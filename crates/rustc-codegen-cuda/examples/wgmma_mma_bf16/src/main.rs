/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only BF16 WGMMA integration example.
//!
//! This example exercises two public-API lowering paths:
//!
//! 1. a single accumulator with a final `wait_group<0>`;
//! 2. two independent accumulator slots with two committed groups,
//!    `wait_group<1>`, and a mandatory final `wait_group<0>`.
//!
//! Both kernels use zero descriptors intentionally. They validate Rust MIR
//! import, WGMMA region selection, LLVM lowering, and PTX generation only.
//! Do not launch either kernel.
//!
//! Usage:
//!   cargo oxide build wgmma_mma_bf16 --arch sm_90a

use cuda_device::wgmma::{
    wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16, wgmma_wait_group,
};
use cuda_device::{DisjointSlice, kernel, thread};

/// Compile-only full-drain WGMMA kernel.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
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

/// Compile-only two-slot partial-wait WGMMA kernel.
///
/// Two independent accumulators allow two committed groups to coexist. The
/// partial wait leaves at most one group pending; the final full wait makes both
/// accumulators safe to observe.
///
/// # Safety
///
/// The zero descriptors are not valid WGMMA shared-memory descriptors. This
/// kernel must not be executed.
#[kernel]
pub unsafe fn wgmma_partial_wait_kernel(mut out: DisjointSlice<u32>) {
    let mut acc0: [[f32; 8]; 4] = [[0.0f32; 8]; 4];
    let mut acc1: [[f32; 8]; 4] = [[0.0f32; 8]; 4];

    unsafe {
        wgmma_fence();

        wgmma_mma_m64n64k16_f32_bf16(&mut acc0, 0u64, 0u64);
        wgmma_commit_group();

        wgmma_mma_m64n64k16_f32_bf16(&mut acc1, 0u64, 0u64);
        wgmma_commit_group();

        wgmma_wait_group::<1>();
        wgmma_wait_group::<0>();
    }

    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc0[0][0].to_bits() ^ acc1[0][0].to_bits();
    }
}

fn main() {
    println!("SUCCESS: BF16 WGMMA value-threaded and partial-wait lowering compiled.");
}
