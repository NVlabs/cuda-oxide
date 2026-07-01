/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::device;

#[device]
pub fn helper_has_no_host_launch_proof() {
    let _ = cuda_device::thread::index_1d32();
}

fn main() {}
