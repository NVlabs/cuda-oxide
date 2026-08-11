/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for DST `size_of_val` / `align_of_val` lowering.
//!
//! The slice case exercises runtime fat-pointer length metadata with a
//! non-byte element type. The `str` case reinterprets a runtime UTF-8 byte
//! slice as `str`, preserving the same `(data, byte_len)` fat-pointer layout.
//!
//! Usage:
//!   cargo oxide run dst_metadata

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Report `size_of_val` and `align_of_val` for a runtime `&[u32]`.
    #[kernel]
    pub fn slice_metadata(input: &[u32], mut out: DisjointSlice<usize>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        let size = core::mem::size_of_val(input);
        let align = core::mem::align_of_val(input);

        unsafe {
            let out_ptr = out.as_mut_ptr();
            out_ptr.write(size);
            out_ptr.add(1).write(align);
        }
    }

    /// Report DST layout for a runtime `str` built from UTF-8 bytes.
    #[kernel]
    pub fn str_metadata(input: &[u8], mut out: DisjointSlice<usize>) {
        if thread::index_1d().get() != 0 {
            return;
        }

        // SAFETY: the host passes the UTF-8 bytes of a Rust string literal.
        let text = unsafe { core::str::from_utf8_unchecked(input) };
        let size = core::mem::size_of_val(text);
        let align = core::mem::align_of_val(text);

        unsafe {
            let out_ptr = out.as_mut_ptr();
            out_ptr.write(size);
            out_ptr.add(1).write(align);
        }
    }
}

fn main() {
    println!("=== dst_metadata ===");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(1);

    let slice_input = [11_u32, 22, 33, 44, 55, 66, 77];
    let device_slice = DeviceBuffer::from_host(&stream, &slice_input).unwrap();
    let mut slice_out = DeviceBuffer::<usize>::zeroed(&stream, 2).unwrap();

    // SAFETY: one thread executes the metadata query and the two-element output
    // buffer covers both stores.
    unsafe { module.slice_metadata(&stream, cfg, &device_slice, &mut slice_out) }
        .expect("slice_metadata launch");

    let slice_result = slice_out.to_host_vec(&stream).unwrap();
    assert_eq!(
        slice_result[0],
        slice_input.len() * core::mem::size_of::<u32>(),
        "size_of_val(&[u32])"
    );
    assert_eq!(
        slice_result[1],
        core::mem::align_of::<u32>(),
        "align_of_val(&[u32])"
    );
    println!("PASS: size_of_val on &[u32]");
    println!("PASS: align_of_val on &[u32]");

    // `oxide` is five bytes and `✓` is three UTF-8 bytes. This deliberately
    // distinguishes str byte-length metadata (8) from character count (6).
    let text_bytes = "oxide✓".as_bytes();
    let device_text = DeviceBuffer::from_host(&stream, text_bytes).unwrap();
    let mut str_out = DeviceBuffer::<usize>::zeroed(&stream, 2).unwrap();

    // SAFETY: `text_bytes` is valid UTF-8 and the output has two elements.
    unsafe { module.str_metadata(&stream, cfg, &device_text, &mut str_out) }
        .expect("str_metadata launch");

    let str_result = str_out.to_host_vec(&stream).unwrap();
    assert_eq!(str_result[0], text_bytes.len(), "size_of_val(str)");
    assert_eq!(
        str_result[1],
        core::mem::align_of::<u8>(),
        "align_of_val(str)"
    );
    println!("PASS: size_of_val on str");
    println!("PASS: align_of_val on str");
    println!("PASS: dst_metadata");
}
