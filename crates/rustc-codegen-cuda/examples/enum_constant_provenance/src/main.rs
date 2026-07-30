/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime regression for pointer provenance in enum constants.
//!
//! Covers niche-encoded enums pointing to a device static or an interior
//! static subobject, plus direct-tagged enums.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::kernel;
use cuda_host::cuda_module;

const FIRST: u64 = 0x1122_3344_5566_7788;
const SECOND: u64 = 0x8877_6655_4433_2211;

static TARGETS: [u64; 2] = [FIRST, SECOND];

const NICHE_STATIC: Option<&'static u64> = Some(&TARGETS[1]);
const NICHE_NONE: Option<&'static u64> = None;

#[repr(u8)]
#[derive(Clone, Copy)]
enum TaggedReference {
    Empty,
    Present(&'static u64),
}

const DIRECT_EMPTY: TaggedReference = TaggedReference::Empty;
const DIRECT_TAGGED: TaggedReference = TaggedReference::Present(&TARGETS[0]);

#[inline(never)]
fn niche_static() -> Option<&'static u64> {
    NICHE_STATIC
}

#[inline(never)]
fn niche_none() -> Option<&'static u64> {
    NICHE_NONE
}

#[inline(never)]
fn direct_empty() -> TaggedReference {
    DIRECT_EMPTY
}

#[inline(never)]
fn direct_tagged() -> TaggedReference {
    DIRECT_TAGGED
}

#[inline(never)]
fn tagged_value(value: TaggedReference) -> u64 {
    match value {
        TaggedReference::Empty => 0,
        TaggedReference::Present(pointer) => *pointer,
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `output` must point to writable device memory for four `u64` values.
    #[kernel]
    pub unsafe fn enum_pointer_constants(output: *mut u64) {
        let static_value = niche_static().map_or(0, |pointer| *pointer);
        let none_value = niche_none().map_or(0, |pointer| *pointer);
        let direct_empty_value = tagged_value(direct_empty());
        let direct_tagged_value = tagged_value(direct_tagged());

        unsafe {
            output.add(0).write(static_value);
            output.add(1).write(none_value);
            output.add(2).write(direct_empty_value);
            output.add(3).write(direct_tagged_value);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const OUTPUT_COUNT: usize = 4;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;
    let output = DeviceBuffer::<u64>::zeroed(&stream, OUTPUT_COUNT)?;

    // SAFETY: the output allocation contains four u64 values and exactly one
    // thread is launched.
    unsafe {
        module.enum_pointer_constants(
            &stream,
            LaunchConfig::for_num_elems(1),
            output.cu_deviceptr() as *mut u64,
        )
    }?;

    let actual = output.to_host_vec(&stream)?;
    let expected = [SECOND, 0, 0, FIRST];

    assert_eq!(
        actual.as_slice(),
        expected.as_slice(),
        "enum constant pointer provenance produced incorrect GPU output"
    );

    println!("enum_constant_provenance: PASS");
    Ok(())
}
