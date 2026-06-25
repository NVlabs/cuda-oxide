// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use core::num::NonZeroU32;

use cuda_core::DeviceCopy;

#[repr(transparent)]
#[derive(Copy, Clone)]
struct ZeroInvalid(NonZeroU32);

// SAFETY: This local impl is deliberately used to isolate enum zero-validity:
// the derive must reject `BadPayload` because zero is invalid for the payload,
// not because the field fails the ordinary `DeviceCopy` bound.
unsafe impl DeviceCopy for ZeroInvalid {}

#[derive(Copy, Clone, DeviceCopy)]
enum BadPayload {
    Scalar(ZeroInvalid),
}

fn main() {
    let _ = core::mem::size_of::<BadPayload>();
}
