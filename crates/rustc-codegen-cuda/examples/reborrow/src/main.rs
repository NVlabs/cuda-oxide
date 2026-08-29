/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression test for `Rvalue::Reborrow` translation in device code.
//!
//! `feature(reborrow)` (rust-lang/rust#145612) lets a user ADT that
//! implements the `core::marker::Reborrow` trait be passed by value
//! repeatedly: each use site gets an implicit reborrow instead of a move.
//! Since nightly-2026-08-28 those implicit reborrows appear in MIR as a
//! dedicated `Rvalue::Reborrow(Ty, Mutability, Place)`. The release MIR
//! pipeline (GVN/copy-prop) folds these into plain copies before the
//! importer sees them, so the smoketest runs this example with
//! `--device-debug` (the -Zmir-opt-level=0 device path), where the rvalue
//! reaches the importer intact (verified by stubbing the arm: the stub
//! fails this example under --device-debug and not under release).
//!
//! The kernel below passes a `Reborrow` wrapper around `&mut f32` to a
//! helper twice; without the importer's `Rvalue::Reborrow` arm this fails
//! device codegen with an "Unsupported construct" error.

#![feature(reborrow)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Reborrowable exclusive view of one element. Exactly one lifetime
    /// parameter, as the `Reborrow` trait requires.
    pub struct MutView<'a> {
        data: &'a mut f32,
    }

    impl<'a> core::marker::Reborrow for MutView<'a> {}

    fn add_one(v: MutView<'_>) {
        *v.data += 1.0;
    }

    #[kernel]
    pub fn reborrow(mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(elem) = c.get_mut(idx) {
            let v = MutView { data: elem };
            // Each call takes `v` by value; the compiler inserts an implicit
            // reborrow (`Rvalue::Reborrow`) so `v` stays live for the second
            // call. That second call is the proof the reborrow happened.
            add_one(v);
            add_one(v);
        }
    }
}

fn main() {
    println!("=== Rvalue::Reborrow device regression ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: launch shape/resources match the kernel; the buffer covers its
    // accesses.
    unsafe { module.reborrow(&stream, LaunchConfig::for_num_elems(N as u32), &mut c_dev) }
        .expect("Kernel launch failed");

    let c_host = c_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for (i, &value) in c_host.iter().enumerate() {
        if (value - 2.0).abs() > 1e-6 {
            if errors < 5 {
                eprintln!("  Error at [{i}]: expected 2.0, got {value}");
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ SUCCESS: all {N} elements were reborrow-incremented twice");
    } else {
        println!("✗ FAILED: {errors} errors");
        std::process::exit(1);
    }
}
