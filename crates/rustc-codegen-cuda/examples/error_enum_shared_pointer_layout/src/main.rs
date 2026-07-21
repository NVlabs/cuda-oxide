/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: enum storage containing a shared-memory pointer must fail
//! closed until MIR lowering knows which exporter data layout was selected.

use cuda_device::{SharedArray, kernel};

static mut SHARED: SharedArray<u32, 1> = SharedArray::UNINIT;

#[inline(never)]
fn pointer_bits(value: Option<&'static SharedArray<u32, 1>>) -> u64 {
    value.map_or(0, |pointer| pointer as *const SharedArray<u32, 1> as u64)
}

/// # Safety
///
/// `out` must point to writable device memory for one `u64`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn shared_pointer_enum(out: *mut u64) {
    let shared: &'static SharedArray<u32, 1> = unsafe { &*(&raw const SHARED) };
    unsafe {
        *out = pointer_bits(Some(shared));
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
