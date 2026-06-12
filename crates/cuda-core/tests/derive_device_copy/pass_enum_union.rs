// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

#[derive(Copy, Clone, DeviceCopy)]
enum Packet {
    Empty,
    Scalar(u32),
    Pair { x: f32, y: [u16; 2] },
}

#[derive(Copy, Clone, DeviceCopy)]
union Word {
    bits: u32,
    scalar: f32,
}

fn assert_device_copy<T: DeviceCopy>() {}

fn main() {
    assert_device_copy::<Packet>();
    assert_device_copy::<Word>();
}
