// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_core::DeviceCopy;

#[derive(Copy, Clone, DeviceCopy)]
#[repr(u8)]
enum Tag {
    Zero = 0,
    One = 1,
}

#[derive(Copy, Clone, DeviceCopy)]
enum Packet {
    Empty,
    Scalar(u32),
    Pair { x: f32, y: [u16; 2] },
}

fn assert_device_copy<T: DeviceCopy>() {}

fn main() {
    assert_device_copy::<Tag>();
    assert_device_copy::<Packet>();
}
