/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: an interior pointer into a device static whose pointee is
//! unsized (a slice) is rejected with a clean diagnostic.
//!
//! Usage:
//!   cargo oxide build error_static_slice_addend
//!
//! Expected: the build fails explaining that the slice pointee needs
//! fat-pointer metadata that cuda-oxide does not yet preserve.

use cuda_device::kernel;

static TABLE: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];

/// `&TABLE[2]` is `&[f32; 2]`; the unsize coercion to `&[f32]` keeps the
/// 16-byte addend selecting element 2 but adds a length that the thin
/// device pointer emitted for interior-static addends cannot carry.
const PAIR_SLICE: &'static [f32] = &TABLE[2];

#[inline(never)]
fn pair_slice() -> &'static [f32] {
    PAIR_SLICE
}

/// This must fail during import. Emitting the constant as a thin pointer
/// typed as a slice would drop the length metadata and misread the
/// fat-pointer layout downstream.
///
/// # Safety
///
/// `out` must point to device-accessible storage that is properly aligned
/// and writable for one `f32`. No other thread may race with this write.
#[kernel]
pub unsafe fn slice_addend(out: *mut f32) {
    let pair = pair_slice();
    unsafe {
        *out = pair[0] + pair[1];
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
