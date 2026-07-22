/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for MIR import of array constants.
//!
//! Covered shapes:
//! - bare `[T; N]` constants indexed by a runtime value,
//! - nested `[[T; M]; N]` constants,
//! - arrays of padded tuple constants containing no-payload enums,
//! - nested tuples with zero-sized fields,
//! - non-empty all-ZST tuples whose fields have equal offsets,
//! - tuple arrays whose fields rustc reorders in memory,
//! - tuple arrays containing an over-aligned zero-sized field,
//! - direct padded tuple constants,
//! - pointer-to-array constants (`&[T; N]`), which predate bare-array support.
//!
//! Run with:
//!   cargo oxide run array_constants
//!   ./crates/rustc-codegen-cuda/examples/array_constants/verify-code-shape.sh

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

const BARE_TABLE: [f32; 4] = [1.25, -2.5, 5.0, 10.5];
const NESTED_TABLE: [[u32; 3]; 2] = [[11, 13, 17], [19, 23, 29]];
const POINTER_TABLE: &[u32; 4] = &[31, 37, 41, 43];
const TUPLE_TABLE: [(bool, Side); 6] = [
    (false, Side::LowX),
    (true, Side::HighX),
    (false, Side::LowY),
    (true, Side::HighY),
    (false, Side::LowZ),
    (true, Side::HighZ),
];
const NESTED_TUPLE_TABLE: [((u8, ()), u32); 2] = [((3, ()), 17), ((5, ()), 29)];
const ALL_ZST_TUPLE_TABLE: [(((), ()), u32); 2] = [(((), ()), 59), (((), ()), 61)];
// rustc lays these fields out at offsets 4, 0, and 8 respectively on the
// supported 64-bit target. Reading the allocation in declaration order would
// therefore corrupt both of the first two values.
const REORDERED_TUPLE_TABLE: [(u8, u32, u64); 2] = [
    (0xa5, 0x1122_3344, 0x0102_0304_0506_0708),
    (0x5a, 0x99aa_bbcc, 0x8877_6655_4433_2211),
];
#[derive(Clone, Copy)]
#[repr(align(32))]
struct Align32;

const OVERALIGNED_ZST_TUPLE_TABLE: [(Align32, u8); 2] = [(Align32, 0x12), (Align32, 0x34)];
const DIRECT_TUPLE: (u8, u32) = (7, 41);

#[derive(Clone, Copy)]
#[repr(u32)]
enum Side {
    LowX = 1,
    HighX = 2,
    LowY = 3,
    HighY = 4,
    LowZ = 5,
    HighZ = 6,
}

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

    #[inline(never)]
    fn tuple_array_value(i: usize) -> u32 {
        let (is_high, side) = TUPLE_TABLE[i % 6];
        (side as u32) * 10 + (is_high as u32)
    }

    #[inline(never)]
    fn nested_tuple_array_value(i: usize) -> u32 {
        let ((tag, ()), value) = NESTED_TUPLE_TABLE[i & 1];
        tag as u32 + value
    }

    #[inline(never)]
    fn all_zst_tuple_array_value(i: usize) -> u32 {
        let (((), ()), value) = ALL_ZST_TUPLE_TABLE[i & 1];
        value
    }

    #[inline(never)]
    fn reordered_tuple_array_value(i: usize) -> u32 {
        let (byte, word, wide) = REORDERED_TUPLE_TABLE[i & 1];
        (byte as u32)
            .wrapping_mul(257)
            .wrapping_add(word)
            .wrapping_mul(257)
            .wrapping_add(wide as u32)
            .wrapping_mul(257)
            .wrapping_add((wide >> 32) as u32)
    }

    #[inline(never)]
    fn overaligned_zst_tuple_array_value(i: usize) -> u32 {
        let pair = OVERALIGNED_ZST_TUPLE_TABLE[i & 1];
        let address_low_bits = (&pair as *const (Align32, u8) as usize) & 31;
        let (_, byte) = pair;
        byte as u32 + address_low_bits as u32
    }

    #[inline(never)]
    fn direct_tuple_value() -> (u8, u32) {
        DIRECT_TUPLE
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
            let tuple = tuple_array_value(i);
            let nested_tuple = nested_tuple_array_value(i);
            let (direct_tag, direct_value) = direct_tuple_value();
            let direct = direct_tag as u32 + direct_value;
            let all_zst = all_zst_tuple_array_value(i);
            let reordered = reordered_tuple_array_value(i);
            let overaligned_zst = overaligned_zst_tuple_array_value(i);
            *slot = nested
                .wrapping_mul(257)
                .wrapping_add(pointer)
                .wrapping_mul(257)
                .wrapping_add(tuple)
                .wrapping_mul(257)
                .wrapping_add(nested_tuple)
                .wrapping_mul(257)
                .wrapping_add(direct)
                .wrapping_mul(257)
                .wrapping_add(all_zst)
                .wrapping_mul(257)
                .wrapping_add(reordered)
                .wrapping_mul(257)
                .wrapping_add(overaligned_zst);
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
    let (is_high, side) = TUPLE_TABLE[i % 6];
    let tuple = (side as u32) * 10 + (is_high as u32);
    let ((tag, ()), value) = NESTED_TUPLE_TABLE[i & 1];
    let nested_tuple = tag as u32 + value;
    let (direct_tag, direct_value) = DIRECT_TUPLE;
    let direct = direct_tag as u32 + direct_value;
    let (((), ()), all_zst) = ALL_ZST_TUPLE_TABLE[i & 1];
    let (byte, word, wide) = REORDERED_TUPLE_TABLE[i & 1];
    let reordered = (byte as u32)
        .wrapping_mul(257)
        .wrapping_add(word)
        .wrapping_mul(257)
        .wrapping_add(wide as u32)
        .wrapping_mul(257)
        .wrapping_add((wide >> 32) as u32);
    let (_, overaligned_zst) = OVERALIGNED_ZST_TUPLE_TABLE[i & 1];
    nested
        .wrapping_mul(257)
        .wrapping_add(pointer)
        .wrapping_mul(257)
        .wrapping_add(tuple)
        .wrapping_mul(257)
        .wrapping_add(nested_tuple)
        .wrapping_mul(257)
        .wrapping_add(direct)
        .wrapping_mul(257)
        .wrapping_add(all_zst)
        .wrapping_mul(257)
        .wrapping_add(reordered)
        .wrapping_mul(257)
        .wrapping_add(overaligned_zst as u32)
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
        println!(
            "array_constants: PASS ({N} threads; primitive, padded/reordered/over-aligned tuple, nested/equal-offset ZST tuple, pointer-to-array constants)"
        );
        Ok(())
    } else {
        println!("array_constants: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
