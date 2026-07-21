/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: enum constants containing pointer relocations must fail
//! closed until the importer can preserve those relocations.

use cuda_device::kernel;

static TARGET: u64 = 0x1122_3344_5566_7788;
const POINTER_ENUM: Option<&'static u64> = Some(&TARGET);

#[inline(never)]
fn pointer_enum() -> Option<&'static u64> {
    POINTER_ENUM
}

/// # Safety
///
/// `out` must point to writable device memory for one `u64`, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn enum_pointer_constant(out: *mut u64) {
    let address = pointer_enum().map_or(0, |pointer| pointer as *const u64 as u64);
    unsafe {
        *out = address;
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
