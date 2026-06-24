/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for the `#[unroll]` / `#[unroll(N)]` loop-unroll transform.
//!
//! The attribute goes directly on the loop it should unroll (the `#[kernel]`
//! macro reads it and tags that loop): a function can unroll one loop and leave
//! its neighbours alone.
//!
//! `full_unroll` has a compile-time-constant trip count, so `#[unroll]` should
//! unroll it completely and the per-iteration `i & 3` should fold to literals.
//! `partial_unroll` has a runtime trip count, so `#[unroll(4)]` unrolls the body
//! by 4 and leaves a remainder loop. Both are semantics-preserving; the host
//! checks the sums.
//!
//! Run: cargo oxide run unroll_smoke

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Full unroll of a constant-trip-count loop. `acc` starts at the thread
    /// index and adds `i & 3` for `i` in `0..8` (= 0+1+2+3+0+1+2+3 = 12), so
    /// `out[tid] == tid + 12`.
    #[kernel]
    pub fn full_unroll(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            #[unroll]
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
    pub fn partial_unroll(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) of a runtime loop whose body uses `i & 3` (the
    /// gemm "stage" pattern). After unrolling, the main loop's counter is a
    /// multiple of 4, so `(i+j) & 3` should fold to the constants `0,1,2,3`.
    /// `out[tid]` is the sum of `i & 3` for `i` in `0..n`.
    #[kernel]
    pub fn partial_fold(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                acc = acc.wrapping_add(i & 3);
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Full unroll of a loop whose body has **internal control flow** (an
    /// `if`/`else`), so the body is several basic blocks, not one. This is the
    /// case the earlier single-block unroller could not handle. For `i` in
    /// `0..8`: even `i` adds `i` (0+2+4+6 = 12), odd `i` adds 10 (4 * 10 = 40),
    /// so `out[tid] == tid + 52`.
    #[kernel]
    pub fn full_mb(mut out: DisjointSlice<u32>) {
        let tid = thread::index_1d();
        let base = tid.get() as u32;
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = base;
            let mut i: u32 = 0;
            #[unroll]
            while i < 8 {
                if i & 1 == 0 {
                    acc = acc.wrapping_add(i);
                } else {
                    acc = acc.wrapping_add(10);
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Partial unroll (by 4) of a runtime loop whose body has internal control
    /// flow. For `i` in `0..n`: even `i` adds `i`, odd `i` adds 100. With n=10:
    /// even (0+2+4+6+8 = 20) + odd (5 * 100 = 500) = 520.
    #[kernel]
    pub fn partial_mb(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            #[unroll(4)]
            while i < n {
                if i & 1 == 0 {
                    acc = acc.wrapping_add(i);
                } else {
                    acc = acc.wrapping_add(100);
                }
                i += 1;
            }
            *out_elem = acc;
        }
    }

    /// Regression guard: the loop bound `hi` is **loop-carried** (it changes each
    /// iteration), so partial unroll's "does a group of 4 still fit" guard would
    /// be unsound. The pass must refuse this loop (a loud warning) and leave it as
    /// an ordinary loop, NOT miscompile or crash. For n=10 it runs 5 iterations
    /// (i=0..4 before i meets the shrinking hi), so `out[tid] == 0+1+2+3+4 = 10`.
    #[kernel]
    pub fn carried_bound(mut out: DisjointSlice<u32>, n: u32) {
        let tid = thread::index_1d();
        if let Some(out_elem) = out.get_mut(tid) {
            let mut acc: u32 = 0;
            let mut i: u32 = 0;
            let mut hi: u32 = n;
            #[unroll(4)]
            while i < hi {
                acc = acc.wrapping_add(i);
                i += 1;
                hi = hi.wrapping_sub(1);
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

    let mut d_fold = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .partial_fold(stream.as_ref(), cfg, &mut d_fold, trip)
        .expect("launch partial_fold");
    let got_fold = d_fold.to_host_vec(&stream).unwrap();

    let mut d_fullmb = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .full_mb(stream.as_ref(), cfg, &mut d_fullmb)
        .expect("launch full_mb");
    let got_fullmb = d_fullmb.to_host_vec(&stream).unwrap();

    let mut d_partmb = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .partial_mb(stream.as_ref(), cfg, &mut d_partmb, trip)
        .expect("launch partial_mb");
    let got_partmb = d_partmb.to_host_vec(&stream).unwrap();

    let mut d_carried = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    module
        .carried_bound(stream.as_ref(), cfg, &mut d_carried, trip)
        .expect("launch carried_bound");
    let got_carried = d_carried.to_host_vec(&stream).unwrap();

    let mut failures = 0usize;
    let want_part = trip * (trip - 1) / 2;
    let want_fold: u32 = (0..trip).map(|i| i & 3).sum();
    let want_partmb: u32 = (0..trip).map(|i| if i & 1 == 0 { i } else { 100 }).sum();
    let want_carried: u32 = {
        let (mut a, mut i, mut hi) = (0u32, 0u32, trip);
        while i < hi {
            a = a.wrapping_add(i);
            i += 1;
            hi = hi.wrapping_sub(1);
        }
        a
    };
    for tid in 0..N {
        let want_full = tid as u32 + 12;
        let want_fullmb = tid as u32 + 52;
        if got_full[tid] != want_full {
            println!("FAIL tid={tid}: full_unroll={} expected={want_full}", got_full[tid]);
            failures += 1;
        }
        if got_part[tid] != want_part {
            println!("FAIL tid={tid}: partial_unroll={} expected={want_part}", got_part[tid]);
            failures += 1;
        }
        if got_fold[tid] != want_fold {
            println!("FAIL tid={tid}: partial_fold={} expected={want_fold}", got_fold[tid]);
            failures += 1;
        }
        if got_fullmb[tid] != want_fullmb {
            println!("FAIL tid={tid}: full_mb={} expected={want_fullmb}", got_fullmb[tid]);
            failures += 1;
        }
        if got_partmb[tid] != want_partmb {
            println!("FAIL tid={tid}: partial_mb={} expected={want_partmb}", got_partmb[tid]);
            failures += 1;
        }
        if got_carried[tid] != want_carried {
            println!("FAIL tid={tid}: carried_bound={} expected={want_carried}", got_carried[tid]);
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
