/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Struct constant with a thin pointer field to a device static.
//!
//! Aggregate const values materialize each thin pointer field via
//! `MirGlobalAllocOp`; the kernel reads through that pointer and a sibling
//! scalar field.
//!
//! Run: `cargo oxide run struct_constant_provenance`

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{kernel, thread};
use cuda_host::cuda_module;

static FIRST: [u8; 16] = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

pub struct Holder {
    pub pointer: &'static [u8; 16],
    pub flag: bool,
}

const DIRECT: Holder = Holder {
    pointer: &FIRST,
    flag: true,
};

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `output` must point to writable device-accessible storage for one `u8` per
    /// launched thread.
    #[kernel]
    pub unsafe fn direct_struct_pointer(output: *mut u8) {
        let index = thread::index_1d().get();
        let holder = DIRECT;
        unsafe {
            output
                .add(index)
                .write(holder.pointer[index & 15] + holder.flag as u8);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    let out = DeviceBuffer::<u8>::zeroed(&stream, 1).expect("alloc out");
    // SAFETY: one-thread launch writing a single u8.
    unsafe {
        module
            .direct_struct_pointer(
                &stream,
                LaunchConfig::for_num_elems(1),
                out.cu_deviceptr() as *mut u8,
            )
            .expect("launch");
    }
    stream.synchronize().expect("sync");

    let host = out.to_host_vec(&stream).expect("dtoh");
    let expected = FIRST[0] + true as u8;
    assert_eq!(host[0], expected, "got {} expected {}", host[0], expected);
    println!("struct_constant_provenance: PASS ({})", host[0]);
}
