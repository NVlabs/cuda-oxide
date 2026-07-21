/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a nested bool niche must fail closed until aggregate
//! construction explicitly materializes the bool's complete physical byte.

use cuda_device::kernel;

#[repr(C)]
#[derive(Clone, Copy)]
struct Wrapper {
    pad: u32,
    flag: bool,
}

#[inline(never)]
fn make_value(flag: bool) -> Option<Wrapper> {
    Some(Wrapper {
        pad: 0xCAFE_BABE,
        flag,
    })
}

/// # Safety
///
/// `out` must point to two writable device `u32` values, with no racing
/// access from another thread.
#[kernel]
pub unsafe fn nested_bool_niche(flag: bool, out: *mut u32) {
    let value = make_value(flag);
    let (present, payload) = match value {
        Some(wrapper) => (1, wrapper.pad ^ u32::from(wrapper.flag)),
        None => (0, 0),
    };
    unsafe {
        out.write(present);
        out.add(1).write(payload);
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
