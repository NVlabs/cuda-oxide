/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive test: device-side drop glue.
//!
//! Verifies that types with `impl Drop` compile and execute correctly on
//! the device. A `DropMarker` writes a sentinel through a captured pointer
//! when it goes out of scope; the host checks the sentinel after launch.
//!
//! Usage:
//!   cargo oxide run drop_glue
//!
//! Expected: build succeeds, kernel writes 0xDEADBEEF via drop glue.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

pub struct DropMarker {
    target: *mut u32,
}

impl Drop for DropMarker {
    fn drop(&mut self) {
        unsafe {
            self.target.write(0xDEAD_BEEFu32);
        }
    }
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn drop_glue_kernel(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(slot) = out.get_mut(idx) {
            *slot = 0;
            let _m = DropMarker {
                target: slot as *mut u32,
            };
            // _m drops at end of scope; expected to write 0xDEADBEEF.
        }
    }
}

fn main() {
    println!("=== drop_glue ===\n");

    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");

    const N: usize = 256;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    // SAFETY: launch shape/resources match the kernel; buffer covers its accesses.
    unsafe {
        module
            .drop_glue_kernel(&stream, LaunchConfig::for_num_elems(N as u32), &mut out_dev)
            .expect("drop_glue_kernel launch");
    }

    let out = out_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0usize;
    for (i, &val) in out.iter().enumerate() {
        if val != 0xDEAD_BEEF {
            if errors < 5 {
                eprintln!(
                    "  FAIL drop_glue[{}]: got {:#010X} expected 0xDEADBEEF",
                    i, val
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("SUCCESS: drop glue wrote sentinel in all {} elements", N);
    } else {
        eprintln!("FAIL: {} errors", errors);
        std::process::exit(1);
    }
}
