/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Static shared-memory access through `llvm.addressof` (guards issue #54).
//!
//! The kernel does `OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight` on a static
//! `SharedArray<f32, 1>`. Before the fix in PR #55, the llvm-export textual exporter
//! gave the `addressof @__shared_mem_N` result a `%vN` SSA name even though
//! `addressof` is virtual in textual LLVM IR (it has no instruction form,
//! only a symbol reference at use sites). When the use printed before the
//! addressof's block, the GEP referenced a `%vN` no instruction defined and
//! libNVVM rejected the IR.
//!
//! This example launches the kernel through `cuda_host::ltoir::load_kernel_module`,
//! which compiles the cuda-oxide-emitted NVVM IR via libNVVM and links the
//! cubin via nvJitLink. A dangling SSA reference in the `.ll` would fail at
//! libNVVM's verifier before the kernel could run, so a regression of #54
//! is now a hard runtime failure instead of a silent build artifact.
//!
//! Run: `cargo oxide run addressof_sharedarray`

#![allow(static_mut_refs)]
#![allow(clippy::assign_op_pattern)] // Expanded assignment preserves the addressof repro CFG.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, device, kernel, thread};
use cuda_host::{cuda_module, ltoir};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn sharedarray_late_use(seed: f32, mut out: DisjointSlice<f32>) {
        static mut OUTPUT_NORM: SharedArray<f32, 1> = SharedArray::UNINIT;

        if thread::index_1d().get() == 0 {
            unsafe {
                OUTPUT_NORM[0] = seed;
                let weight = repro_weight();
                // Issue #54 repro shape: load addressof[0], multiply, store.
                OUTPUT_NORM[0] = OUTPUT_NORM[0] * weight;
                *out.get_unchecked_mut(0) = OUTPUT_NORM[0];
            }
        }
    }

    #[inline(never)]
    #[device]
    fn repro_weight() -> f32 {
        3.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== addressof_sharedarray (issue #54 regression) ===");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    // Forces the cuda-oxide-emitted `.ll` through libNVVM + nvJitLink.
    // A dangling SSA reference in the IR would fail libNVVM's verifier here.
    let raw_module = ltoir::load_kernel_module(&ctx, "addressof_sharedarray")?;
    let module = kernels::from_module(raw_module).expect("typed module init failed");

    let cfg = LaunchConfig::for_num_elems(1);
    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1)?;
    let seed: f32 = 7.0;

    // SAFETY: one thread is launched, matching the kernel's single-element
    // output access and fixed shared-memory use.
    unsafe { module.sharedarray_late_use(stream.as_ref(), cfg, seed, &mut out) }?;

    let result = out.to_host_vec(&stream)?[0];
    let expected: f32 = 21.0; // seed * repro_weight() == 7.0 * 3.0

    if (result - expected).abs() < f32::EPSILON {
        println!("PASS addressof_sharedarray: seed={seed}, result={result}");
        Ok(())
    } else {
        eprintln!("FAIL addressof_sharedarray: got {result}, expected {expected}");
        std::process::exit(1);
    }
}
