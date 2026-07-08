/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Device-only crate exercising `#[link_name = "llvm.*"]` intrinsic externs
//! (the `link_llvm_intrinsics` pattern: declaring an LLVM intrinsic directly
//! as a foreign item instead of going through a wrapper API).
#![no_std]
#![feature(link_llvm_intrinsics)]
#![allow(internal_features)]

use cuda_device::kernel;

// The Rust-side names are arbitrary — dispatch keys on the link symbol, so
// the wrappers can use familiar CUDA names.
unsafe extern "C" {
    #[link_name = "llvm.nvvm.read.ptx.sreg.tid.x"]
    fn thread_idx_x() -> u32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ntid.x"]
    fn block_dim_x() -> u32;
    #[link_name = "llvm.nvvm.read.ptx.sreg.ctaid.x"]
    fn block_idx_x() -> u32;
    #[link_name = "llvm.nvvm.barrier0"]
    fn sync_threads();
}

#[kernel]
pub fn linkname_ids(out: &mut [u32]) {
    let i = unsafe { (block_idx_x() * block_dim_x() + thread_idx_x()) as usize };
    unsafe { sync_threads() };
    if i < out.len() {
        out[i] = i as u32 * 3 + 1;
    }
}
