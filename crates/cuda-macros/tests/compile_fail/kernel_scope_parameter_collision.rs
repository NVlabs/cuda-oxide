// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::kernel;

#[kernel(scope = launch)]
pub fn collision(launch: u32) {
    let _ = launch;
}

fn main() {}
