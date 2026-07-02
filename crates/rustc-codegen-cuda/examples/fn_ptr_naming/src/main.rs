/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Regression for issue #130: naming a fn-pointer type inside a kernel.
//!
//! Coercing a named `fn` to `fn(u32) -> u32` makes the type translator
//! see `RigidTy::FnDef` (the zero-sized type of the named function) and
//! `RigidTy::FnPtr` (the pointer it coerces to). Neither had a
//! translation arm, so the kernel failed to import even though nothing
//! is ever CALLED through the pointer.
//!
//! Calling through a fn pointer is still unsupported; this example only
//! pins that the types translate and a reflexive pointer comparison
//! works.
//!
//! Run with:
//!   cargo oxide run fn_ptr_naming

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

fn inc(x: u32) -> u32 {
    x.wrapping_add(1)
}

fn dec(x: u32) -> u32 {
    x.wrapping_sub(1)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fn_ptr_eq(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            // Naming the fn-pointer type is what used to break import.
            let f: fn(u32) -> u32 = inc;
            let g: fn(u32) -> u32 = dec;
            // Same fn compares equal; different fns compare unequal.
            // (Rust permits, but does not promise, distinct fn
            // addresses, so only the == case is guaranteed language-wise.)
            let same = core::ptr::fn_addr_eq(f, f) as u32;
            let diff = (!core::ptr::fn_addr_eq(f, g)) as u32;
            *out_elem = same & diff;
        }
    }
}

fn main() {
    println!("=== fn_ptr_naming regression (issue #130) ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/fn_ptr_naming.ptx");
    let module = ctx
        .load_module_from_file(ptx_path)
        .expect("Failed to load PTX");
    let module = kernels::from_module(module).expect("Failed to initialize typed module");
    let stream = ctx.default_stream();

    const BLOCK: u32 = 32;
    const N: usize = BLOCK as usize;

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut d_out = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.fn_ptr_eq(stream.as_ref(), cfg, &mut d_out) }.expect("launch fn_ptr_eq");
    let got = d_out.to_host_vec(&stream).unwrap();

    let failures = got.iter().filter(|&&g| g != 1).count();
    if failures == 0 {
        println!("fn_ptr_naming: PASS ({N} threads; f == f and f != g everywhere)");
    } else {
        println!("fn_ptr_naming: {failures} FAILURES (expected 1 in every lane)");
        std::process::exit(1);
    }
}
