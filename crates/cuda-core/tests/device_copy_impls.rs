// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::num::Wrapping;

use cuda_core::DeviceCopy;

fn assert_device_copy<T: DeviceCopy>() {}

#[test]
fn device_copy_covers_core_parity_types() {
    assert_device_copy::<bool>();
    assert_device_copy::<char>();

    assert_device_copy::<PhantomData<String>>();
    assert_device_copy::<MaybeUninit<u32>>();
    assert_device_copy::<Wrapping<u64>>();
}
