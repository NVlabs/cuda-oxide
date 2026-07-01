/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: niche-encoded SetDiscriminant remains an explicit support
//! gap until cuda-oxide can write rustc's payload encoding.

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use core::num::NonZeroU32;
use cuda_device::kernel;

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn select_none(value: &mut Option<NonZeroU32>) {
    mir!({
        SetDiscriminant(*value, 0);
        Return()
    })
}

/// Attempts the intentionally unsupported niche discriminant write.
///
/// # Safety
///
/// If `value` is non-null, it must point to a valid, writable
/// `Option<NonZeroU32>` for the duration of the call.
#[kernel]
pub unsafe fn set_discriminant_niche(value: *mut Option<NonZeroU32>) {
    if !value.is_null() {
        unsafe {
            select_none(&mut *value);
        }
    }
}

fn main() {
    println!("This build must fail until niche payload writes are implemented.");
}
