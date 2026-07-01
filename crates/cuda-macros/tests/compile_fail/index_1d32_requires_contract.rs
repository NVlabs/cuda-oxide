// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::{kernel, thread};

#[kernel(scope = scope)]
pub fn missing_contract() {
    let _ = thread::index_1d32(scope);
}

fn main() {}
