/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{Index2D, Runtime2DIndex};
use cuda_device::{device, DisjointSlice};

// A `Runtime2DIndex` witness is minted from the slice whose pitch it uses, so
// it cannot index a slice in a different index space. The pitch travelling
// with the slice is what ties the two together.
#[device]
pub fn bad_runtime_stride(
    pitched: DisjointSlice<u32, Runtime2DIndex>,
    mut out: DisjointSlice<u32, Index2D<100>>,
) {
    let idx = cuda_device::thread::index_2d_runtime(&pitched).unwrap();
    let _ = out.get_mut(idx);
}

fn main() {}
