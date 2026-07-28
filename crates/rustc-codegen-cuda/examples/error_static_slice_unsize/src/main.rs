/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: a zero-addend unsize coercion of a device static to a
//! slice is rejected with a clean diagnostic.
//!
//! Usage:
//!   cargo oxide build error_static_slice_unsize
//!
//! Expected: the build fails explaining that zero-addend pointee
//! reinterpretations and unsized coercions are not supported.

use cuda_device::kernel;

static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// `&TABLE` is `&[f32; 4]`; the unsize coercion to `&[f32]` points at the
/// full static (zero addend) but adds a length that the thin device
/// pointer materialized for static constants cannot carry.
const TABLE_SLICE: &[f32] = &TABLE;

#[inline(never)]
fn table_slice() -> &'static [f32] {
    TABLE_SLICE
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
pub unsafe fn slice_unsize(out: *mut f32) {
    let table = table_slice();
    unsafe {
        *out = table[0] + table[3];
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
