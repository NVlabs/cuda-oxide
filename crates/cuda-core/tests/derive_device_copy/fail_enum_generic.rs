// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

#[derive(Copy, Clone, DeviceCopy)]
enum Maybe<T> {
    Empty,
    Value(T),
}

fn main() {
    let _ = core::mem::size_of::<Maybe<u32>>();
}
