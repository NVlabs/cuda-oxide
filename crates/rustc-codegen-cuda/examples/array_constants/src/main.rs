/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for MIR import of array constants.
//!
//! Covered shapes:
//! - bare `[T; N]` constants indexed by a runtime value,
//! - nested `[[T; M]; N]` constants,
//! - pointer-to-array constants (`&[T; N]`), which predate bare-array support.
//!
//! Run with:
//!   cargo oxide run array_constants

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const BARE_TABLE: [f32; 4] = [1.25, -2.5, 5.0, 10.5];
const NESTED_TABLE: [[u32; 3]; 2] = [[11, 13, 17], [19, 23, 29]];
const POINTER_TABLE: &[u32; 4] = &[31, 37, 41, 43];

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn bare_array_value(i: usize) -> f32 {
        BARE_TABLE[i & 3]
    }

    #[inline(never)]
    fn nested_array_value(i: usize) -> u32 {
        let row = i & 1;
        let col = (i / 2) % 3;
        NESTED_TABLE[row][col]
    }

    #[inline(never)]
    fn pointer_to_array_value(i: usize) -> u32 {
        POINTER_TABLE[i & 3]
    }

    #[kernel]
    pub fn check_array_constants(mut out_f32: DisjointSlice<f32>, mut out_u32: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let i = tid.get();

        if let Some(slot) = out_f32.get_mut(tid) {
            *slot = bare_array_value(i);
        }

        let tid_u32 = thread::index_1d();
        if let Some(slot) = out_u32.get_mut(tid_u32) {
            let nested = nested_array_value(i);
            let pointer = pointer_to_array_value(i);
            *slot = nested * 100 + pointer;
        }
    }
}

fn expected_f32(i: usize) -> f32 {
    BARE_TABLE[i & 3]
}

fn expected_u32(i: usize) -> u32 {
    let row = i & 1;
    let col = (i / 2) % 3;
    let nested = NESTED_TABLE[row][col];
    let pointer = POINTER_TABLE[i & 3];
    nested * 100 + pointer
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== array_constants regression ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    const N: usize = 24;
    let mut out_f32 = DeviceBuffer::<f32>::zeroed(&stream, N)?;
    let mut out_u32 = DeviceBuffer::<u32>::zeroed(&stream, N)?;

    // SAFETY: this is a 1D launch and the kernel bounds-checks each output
    // access against the corresponding slice length.
    unsafe {
        module.check_array_constants(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &mut out_f32,
            &mut out_u32,
        )
    }?;

    let got_f32 = out_f32.to_host_vec(&stream)?;
    let got_u32 = out_u32.to_host_vec(&stream)?;

    let mut failures = 0usize;
    for i in 0..N {
        let want_f32 = expected_f32(i);
        if got_f32[i] != want_f32 {
            println!(
                "FAIL bare array tid={i}: got={} expected={}",
                got_f32[i], want_f32
            );
            failures += 1;
        }

        let want_u32 = expected_u32(i);
        if got_u32[i] != want_u32 {
            println!(
                "FAIL nested/pointer array tid={i}: got={} expected={}",
                got_u32[i], want_u32
            );
            failures += 1;
        }
    }

    if failures == 0 {
        println!("array_constants: PASS ({N} threads; bare, nested, pointer-to-array constants)");
        Ok(())
    } else {
        println!("array_constants: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
