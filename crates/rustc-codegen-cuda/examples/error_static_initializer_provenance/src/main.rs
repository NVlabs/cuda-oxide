/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Negative test: device-global pointer relocations are not yet supported.
//!
//! Usage:
//!   cargo oxide build error_static_initializer_provenance
//!
//! Expected: the build fails with a diagnostic explaining that the `REFERENCE`
//! initializer contains pointer provenance that cuda-oxide cannot yet emit.

use cuda_device::kernel;

static TARGET: u32 = 0x1234_5678;
static REFERENCE: &u32 = &TARGET;

#[inline(never)]
fn reference_slot() -> &'static &'static u32 {
    &REFERENCE
}

/// This must fail during import. Emitting the pointer bytes as ordinary zeros
/// would turn `REFERENCE` into null and silently miscompile the dereference.
///
/// # Safety
///
/// `out` must point to device-accessible storage that is properly aligned and
/// writable for one `u32`. No other thread may race with this write.
#[kernel]
pub unsafe fn pointer_initializer(out: *mut u32) {
    unsafe {
        *out = **reference_slot();
    }
}

fn main() {
    println!("This negative example should fail during device compilation.");
}
