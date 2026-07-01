// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, launch_contract, thread};

#[kernel(scope = scope)]
#[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
pub fn two_dimensional() {
    let _ = thread::index_1d32(scope);
}

fn main() {}
