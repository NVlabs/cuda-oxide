/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired measurement of `warp_reduce`'s two forms (#811 / #1305).
//!
//! One source, two builds: the fast path is selected by the
//! `cuda_oxide_sm_at_least` cfg the device build script derives from the target,
//! so there is no run-time switch to hold constant. That cfg is what #1305
//! adds; on a tree without it both builds take the butterfly and the harness
//! still runs, it just has nothing to compare.
//!
//! ```text
//!   cargo oxide run warp_reduce_redux_bench --arch sm_80   redux.sync
//!   cargo oxide run warp_reduce_redux_bench --arch sm_75   shuffle butterfly
//! ```
//!
//! Two timings per build, because they answer different questions:
//!
//!   * `*_chain` is a dependency chain — each reduction consumes the previous
//!     one — so it measures the latency of one reduction, which is what a
//!     serial scan or a scan-like recurrence pays.
//!   * `*_independent` runs four unrelated reductions per iteration, so the
//!     issue slots and shuffle port have something to overlap. A form that only
//!     wins in the chain is a latency win, not a throughput win.
//!
//! `shuffle_*` is the same butterfly `warp_reduce` uses when the fast path does
//! not apply, written out with `WarpCollective::shfl_xor`. Building at sm_80 and
//! comparing `api_chain` against `shuffle_chain` isolates the reduction from the
//! rest of the build, because both walk the same target and the same compiler.
//!
//! Results are per reduction, amortised over a long kernel, so launch and host
//! overhead are not in the number. The first iterations are checked against a
//! host simulation: a kernel the compiler could fold away is not a timing.

use cuda_device::cooperative_groups::{
    WarpCollective, WarpTile, ops::Sum, this_thread_block, warp_reduce,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// Iterations of the timed loop. Long enough that the kernel dominates its own
/// launch; short enough that the whole example stays interactive.
const ITERATIONS: u32 = 40_000;

/// Iterations used for the correctness pass, where the host can follow along.
const CHECK_ITERATIONS: u32 = 3;

const SAMPLES: usize = 15;
const WARMUPS: usize = 3;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// The shuffle form of an unsigned sum, written out so it can be timed next
    /// to the API on the same target.
    #[inline(always)]
    fn butterfly_sum(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            acc = acc.wrapping_add(warp.shfl_xor(acc, delta));
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_xor(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            acc ^= warp.shfl_xor(acc, delta);
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_max(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            let other = warp.shfl_xor(acc, delta);
            if other > acc {
                acc = other;
            }
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_min(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            let other = warp.shfl_xor(acc, delta);
            if other < acc {
                acc = other;
            }
            delta >>= 1;
        }
        acc
    }

    /// Full-warp sum through the public API, chained.
    #[kernel]
    pub fn api_chain(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let mut v = input[i];
        let mut n = 0;
        while n < iterations {
            v = warp_reduce::<u32, Sum, _>(&warp, v);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = v;
        }
    }

    /// The same chain with the butterfly spelled out.
    #[kernel]
    pub fn shuffle_chain(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let mut v = input[i];
        let mut n = 0;
        while n < iterations {
            v = butterfly_sum(&warp, v);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = v;
        }
    }

    /// Four unrelated reductions per iteration, so latency can be hidden.
    #[kernel]
    pub fn api_independent(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let seed = input[i];
        let (mut a, mut b, mut c, mut d) = (
            seed,
            seed ^ 0x9e37_79b9,
            seed.wrapping_mul(3),
            seed.wrapping_add(0x85eb_ca6b),
        );
        let mut n = 0;
        while n < iterations {
            a = warp_reduce::<u32, Sum, _>(&warp, a);
            b = warp_reduce::<u32, cuda_device::cooperative_groups::ops::BitXor, _>(&warp, b);
            c = warp_reduce::<u32, cuda_device::cooperative_groups::ops::Max, _>(&warp, c);
            d = warp_reduce::<u32, cuda_device::cooperative_groups::ops::Min, _>(&warp, d);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = a ^ b ^ c ^ d;
        }
    }

    /// The same four chains over the butterfly.
    #[kernel]
    pub fn shuffle_independent(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let seed = input[i];
        let (mut a, mut b, mut c, mut d) = (
            seed,
            seed ^ 0x9e37_79b9,
            seed.wrapping_mul(3),
            seed.wrapping_add(0x85eb_ca6b),
        );
        let mut n = 0;
        while n < iterations {
            a = butterfly_sum(&warp, a);
            b = butterfly_xor(&warp, b);
            c = butterfly_max(&warp, c);
            d = butterfly_min(&warp, d);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = a ^ b ^ c ^ d;
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const BLOCK: u32 = 128;
const WARPS: u64 = (BLOCK / 32) as u64;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

fn input(len: usize) -> Vec<u32> {
    let mut state = 0x2468_ace0;
    (0..len).map(|_| lcg(&mut state)).collect()
}

/// Follows `api_chain` / `shuffle_chain`: every lane ends an iteration holding
/// its warp's wrapping sum, so the next iteration reduces 32 copies of it.
fn simulate_chain(values: &[u32], iterations: u32) -> Vec<u32> {
    let mut v = values.to_vec();
    for _ in 0..iterations {
        let mut next = v.clone();
        for warp in 0..v.len() / 32 {
            let base = warp * 32;
            let sum = v[base..base + 32]
                .iter()
                .fold(0u32, |acc, x| acc.wrapping_add(*x));
            next[base..base + 32].fill(sum);
        }
        v = next;
    }
    v
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let len = BLOCK as usize;
    let host_input = input(len);
    let input_dev = DeviceBuffer::from_host(&stream, &host_input).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, len).unwrap();

    // ---- correctness: the timed loop, run short enough to simulate ----
    let mut failed = false;
    for (name, launch) in [
        ("api_chain", 0u8),
        ("shuffle_chain", 1u8),
        ("api_independent", 2u8),
        ("shuffle_independent", 3u8),
    ] {
        // SAFETY: launch shape matches the buffers; every lane stays in range.
        let result = unsafe {
            match launch {
                0 => module.api_chain(stream.as_ref(), cfg, &input_dev, CHECK_ITERATIONS, &mut out_dev),
                1 => module.shuffle_chain(stream.as_ref(), cfg, &input_dev, CHECK_ITERATIONS, &mut out_dev),
                2 => module.api_independent(stream.as_ref(), cfg, &input_dev, CHECK_ITERATIONS, &mut out_dev),
                _ => module.shuffle_independent(stream.as_ref(), cfg, &input_dev, CHECK_ITERATIONS, &mut out_dev),
            }
        };
        result.expect("launch failed");
        let got = out_dev.to_host_vec(&stream).unwrap();
        let paired = if launch < 2 {
            let want = simulate_chain(&host_input, CHECK_ITERATIONS);
            if got == want {
                "matches host".to_string()
            } else {
                failed = true;
                format!("MISMATCH first lane got {:#x} want {:#x}", got[0], want[0])
            }
        } else if got.iter().any(|v| *v != 0) {
            "non-zero".to_string()
        } else {
            failed = true;
            "ALL ZERO".to_string()
        };
        println!("correctness {name}: {paired}");
    }
    if failed {
        std::process::exit(1);
    }

    // ---- timing ----
    //
    // Every sample of every arm is taken in one round-robin pass, so a clock or
    // power drift over the run lands on all four arms instead of on whichever
    // arm happened to be measured last.
    let arms: [(&str, u8, u64); 4] = [
        ("api_chain", 0, 1),
        ("shuffle_chain", 1, 1),
        ("api_independent", 2, 4),
        ("shuffle_independent", 3, 4),
    ];

    let mut launch_arm = |arm: u8| {
        // SAFETY: every arm's buffers cover its own indices; `out_dev` is
        // written by lane `i` only.
        unsafe {
            match arm {
                0 => module.api_chain(stream.as_ref(), cfg, &input_dev, ITERATIONS, &mut out_dev),
                1 => module.shuffle_chain(
                    stream.as_ref(),
                    cfg,
                    &input_dev,
                    ITERATIONS,
                    &mut out_dev,
                ),
                2 => module.api_independent(
                    stream.as_ref(),
                    cfg,
                    &input_dev,
                    ITERATIONS,
                    &mut out_dev,
                ),
                _ => module.shuffle_independent(
                    stream.as_ref(),
                    cfg,
                    &input_dev,
                    ITERATIONS,
                    &mut out_dev,
                ),
            }
        }
        .expect("launch failed");
    };

    for _ in 0..WARMUPS {
        for (_, arm, _) in arms {
            launch_arm(arm);
        }
    }
    stream.synchronize().unwrap();

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(SAMPLES); arms.len()];
    for _ in 0..SAMPLES {
        for (index, (_, arm, _)) in arms.iter().enumerate() {
            let start = Instant::now();
            launch_arm(*arm);
            stream.synchronize().unwrap();
            samples[index].push(start.elapsed().as_secs_f64() * 1e3);
        }
    }

    println!();
    println!("| kernel | median ms | ns per reduction |");
    println!("|---|---|---|");
    for (index, (name, _, factors)) in arms.iter().enumerate() {
        let reductions = u64::from(ITERATIONS) * WARPS * factors;
        let median = median_ms(samples[index].clone());
        let ns = median * 1e6 / reductions as f64;
        println!("| {name} | {median:.3} | {ns:.1} |");
    }

    println!();
    println!("SUCCESS: warp_reduce_redux_bench finished");
}
