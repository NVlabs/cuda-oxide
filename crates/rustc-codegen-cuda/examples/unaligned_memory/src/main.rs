/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression coverage for `core::ptr::read_unaligned` and
//! `core::ptr::write_unaligned`.
//!
//! Both kernels deliberately access a `u32` at `base + 1` of a byte buffer.
//! The resulting pointer is valid for four bytes but is not naturally aligned
//! for `u32`, so ordinary `ptr::read` / `ptr::write` would not be valid.
//!
//! Usage:
//!   cargo oxide run unaligned_memory
//!   CUDA_OXIDE_NO_OPT=1 cargo oxide run unaligned_memory

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn read_unaligned_u32(input: &[u8], mut out: DisjointSlice<u32>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        if let Some(slot) = out.get_mut(thread::index_1d()) {
            unsafe {
                // `input.as_ptr().add(1)` is byte-aligned only. Casting it to
                // `*const u32` deliberately creates an under-aligned pointer.
                let ptr = input.as_ptr().add(1).cast::<u32>();
                *slot = core::ptr::read_unaligned(ptr);
            }
        }
    }

    #[kernel]
    pub fn write_unaligned_u32(value: u32, mut out: DisjointSlice<u8>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        unsafe {
            // Preserve byte 0 and byte 5 as guards. The `u32` write occupies
            // bytes 1..=4 and therefore starts at an address not aligned to 4.
            let ptr = out.as_mut_ptr().add(1).cast::<u32>();
            core::ptr::write_unaligned(ptr, value);
        }
    }
}

fn main() {
    println!("=== unaligned_memory ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    // ---------------------------------------------------------------------
    // read_unaligned
    // ---------------------------------------------------------------------

    const READ_BYTES: [u8; 6] = [0xA5, 0x12, 0x34, 0x56, 0x78, 0x5A];
    let expected_read =
        u32::from_ne_bytes([READ_BYTES[1], READ_BYTES[2], READ_BYTES[3], READ_BYTES[4]]);

    let input = DeviceBuffer::from_host(&stream, &READ_BYTES).expect("read input allocation");
    let mut read_out = DeviceBuffer::<u32>::zeroed(&stream, 1).expect("read output allocation");

    // SAFETY: one thread executes the kernel; `input` contains at least five
    // bytes from the base, and `read_out` contains one writable `u32`.
    unsafe { module.read_unaligned_u32(&stream, cfg, &input, &mut read_out) }
        .expect("read_unaligned_u32 launch");

    let read_result = read_out.to_host_vec(&stream).expect("copy read result");
    assert_eq!(read_result, vec![expected_read], "read_unaligned result");
    println!("PASS: read_unaligned from base + 1");

    // ---------------------------------------------------------------------
    // write_unaligned
    // ---------------------------------------------------------------------

    const WRITE_VALUE: u32 = 0x7856_3412;
    const LEFT_GUARD: u8 = 0xA5;
    const RIGHT_GUARD: u8 = 0x5A;

    let initial_write = [LEFT_GUARD, 0, 0, 0, 0, RIGHT_GUARD];
    let mut write_out =
        DeviceBuffer::from_host(&stream, &initial_write).expect("write output allocation");

    // SAFETY: one thread executes the kernel; `write_out` provides six writable
    // bytes, so the four-byte write beginning at byte 1 is fully in bounds.
    unsafe { module.write_unaligned_u32(&stream, cfg, WRITE_VALUE, &mut write_out) }
        .expect("write_unaligned_u32 launch");

    let write_result = write_out.to_host_vec(&stream).expect("copy write result");
    let expected_bytes = WRITE_VALUE.to_ne_bytes();

    assert_eq!(
        &write_result[1..5],
        &expected_bytes,
        "write_unaligned payload bytes"
    );
    println!("PASS: write_unaligned to base + 1");

    assert_eq!(write_result[0], LEFT_GUARD, "left guard byte");
    assert_eq!(write_result[5], RIGHT_GUARD, "right guard byte");
    println!("PASS: guard bytes preserved");

    println!("PASS: unaligned_memory");
}
