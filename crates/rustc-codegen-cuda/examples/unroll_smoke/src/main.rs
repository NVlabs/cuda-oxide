/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for the `#[unroll]` / `#[unroll(N)]` loop-unroll transform.
//!
//! `full_unroll` has a compile-time-constant trip count, so `#[unroll]` should
//! unroll it completely and the per-iteration `i & 3` should fold to literals.
//! `partial_unroll` has a runtime trip count, so `#[unroll(4)]` unrolls the body
//! by 4 and leaves a remainder loop. Both are semantics-preserving; the host
//! checks the sums.
//!
//! Run: cargo oxide run unroll_smoke

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, unroll};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Full unroll of a constant-trip-count loop. `acc` starts at the thread
    /// index and adds `i & 3` for `i` in `0..8` (= 0+1+2+3+0+1+2+3 = 12), so
    /// `out[tid] == tid + 12`.
    #[kernel]
    #[unroll]
    pub fn full_unroll(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            while i < 8 {
                acc = acc.wrapping_add(i & 3);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) of a runtime-trip-count loop: `out[tid]` is the
    /// sum `0 + 1 + ... + (n-1) == n*(n-1)/2`.
    #[kernel]
    #[unroll(4)]
    pub fn partial_unroll(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            while i < n {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *out_elem = acc;
        }
    }
}

fn main() {
    println!("=== unroll_smoke ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let ptx_path = concat!(env!("CARGO_MANIFEST_DIR"), "/unroll_smoke.ptx");
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

    let mut d_full = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .full_unroll(stream.as_ref(), cfg, &mut d_full)
        .expect("launch full_unroll");
    let got_full = d_full.to_host_vec(&stream).unwrap();

    let trip: u32 = 10;
    let mut d_part = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .partial_unroll(stream.as_ref(), cfg, &mut d_part, trip)
        .expect("launch partial_unroll");
    let got_part = d_part.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    let want_part = trip * (trip - 1) / 2;
    for tid in 0..N {
        let want_full = tid as u32 + 12;
        if got_full[tid] != want_full {
            println!("FAIL tid={tid}: full_unroll={} expected={want_full}", got_full[tid]);
            failures += 1;
        }
        if got_part[tid] != want_part {
            println!("FAIL tid={tid}: partial_unroll={} expected={want_part}", got_part[tid]);
            failures += 1;
        }
    }

    if failures == 0 {
        println!("unroll_smoke: PASS ({N} threads; full + partial unroll correct)");
    } else {
        println!("unroll_smoke: FAIL ({failures} mismatches)");
        std::process::exit(1);
    }
}
