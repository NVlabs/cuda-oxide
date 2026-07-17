/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: `wgmma_mma_*` is not yet implemented.
//!
//! The importer rejects `cuda_device::wgmma::wgmma_mma_*` because sound
//! lowering must preserve delayed 32-register accumulator state across
//! commit and wait. The dialect lowering also rejects the operation as a
//! second guard against silently erasing the multiply-accumulate.
//!
//! Usage:
//!   cargo oxide run error_wgmma_mma_unimplemented
//!
//! Expected: build FAILS with
//!   "WGMMA MMA is not yet supported: lowering must preserve delayed ..."

use cuda_device::wgmma::wgmma_mma_m64n64k16_f32_bf16;
use cuda_device::{DisjointSlice, kernel, thread};

/// # Safety
///
/// This kernel intentionally calls a low-level WGMMA intrinsic with dummy
/// descriptors so the compiler rejects the unsupported lowering before PTX
/// execution is possible.
#[kernel]
pub unsafe fn unsupported_wgmma_mma_kernel(mut out: DisjointSlice<u32>) {
    let mut acc: [[f32; 8]; 4] = [[0.0f32; 8]; 4];
    unsafe {
        wgmma_mma_m64n64k16_f32_bf16(&mut acc, 0u64, 0u64);
    }
    let idx = thread::index_1d();
    if let Some(slot) = out.get_mut(idx) {
        *slot = acc[0][0].to_bits();
    }
}

fn main() {
    println!("=== error_wgmma_mma_unimplemented ===");
    println!("This example is intentionally broken to test the diagnostic for");
    println!("the not-yet-implemented `wgmma.mma_async` lowering.");
    println!();
    println!("If you see this message, the build did NOT fail and the test");
    println!("would have detected the previous silent-miscompile regression.");
}
