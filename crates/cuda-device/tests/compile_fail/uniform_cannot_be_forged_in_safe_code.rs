/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! A launch-uniform witness has no safe constructor. Without this, a kernel
//! could wrap a per-thread value and hand it to `tile_2d32_rt`, whose
//! disjointness argument assumes every thread passes the same stride.

use cuda_device::Uniform;

fn main() {
    let _from_struct_literal = Uniform::<u32> { value: 7 };
}
