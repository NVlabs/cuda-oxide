/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end check that `core::intrinsics::exact_div` works in device code.
//!
//! Before that intrinsic existed, `slice::as_chunks` failed to translate:
//!
//! ```text
//! Translation failed: core::slice::as_chunks::<4>
//!   [core/src/slice/mod.rs:1345:32] Compilation error: invalid input program
//! ```
//!
//! Line 1345 is `exact_div(self.len(), N)`. This exercises both the intrinsic
//! directly and `as_chunks`, the API it was blocking.
//!
//! Run: `cargo oxide run exact_div_chunks --arch sm_86`

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

const N: usize = 256;
const CHUNK: usize = 4;

#[cuda_module]
mod kernels {
    use super::*;

    /// Sums each thread's chunk through `as_chunks`, the safe API `exact_div`
    /// unblocks.
    ///
    /// The weights are distinct so a wrong chunk boundary, or a permutation
    /// inside a chunk, produces a wrong value instead of passing.
    #[kernel]
    pub fn chunk_sum(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        // `as_chunks` computes its chunk count with exact_div(len, CHUNK).
        // This is the call that used to fail to compile.
        let (chunks, _rest) = input.as_chunks::<CHUNK>();
        if let Some(slot) = out.get_mut(idx) {
            if i < chunks.len() {
                let c = chunks[i];
                *slot = c[0] + 2.0 * c[1] + 3.0 * c[2] + 4.0 * c[3];
            } else {
                *slot = -1.0;
            }
        }
    }

    /// Exercises the intrinsic away from `as_chunks`, on a signed and an
    /// unsigned dividend, since the lowering picks `sdiv` or `udiv` from the
    /// operand's signedness.
    #[kernel]
    pub fn exact_div_direct(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(slot) = out.get_mut(idx) {
            // Exact by construction: (i*12)/4 and (i*12)/3.
            let n = (i as u32) * 12;
            let unsigned = n / 4;
            let signed = (n as i32) / 3;
            *slot = unsigned.wrapping_add(signed as u32);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load embedded module");
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // ---- as_chunks path ----
    let input: Vec<f32> = (0..N * CHUNK).map(|i| (i as f32) * 0.5).collect();
    let in_dev = DeviceBuffer::from_host(&stream, &input).unwrap();
    let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel; buffers cover its accesses.
    unsafe { module.chunk_sum(&stream, cfg, &in_dev, &mut out_dev) }.expect("chunk_sum launch");
    let got = out_dev.to_host_vec(&stream).unwrap();

    let mut bad = 0;
    for (i, &g) in got.iter().enumerate().take(N) {
        let c = &input[i * CHUNK..i * CHUNK + CHUNK];
        let want = c[0] + 2.0 * c[1] + 3.0 * c[2] + 4.0 * c[3];
        if (g - want).abs() > 1e-3 {
            if bad < 5 {
                println!("  chunk_sum mismatch at {i}: got {g} want {want}");
            }
            bad += 1;
        }
    }
    println!("as_chunks::<4>  : {} / {N} correct", N - bad);

    // ---- direct intrinsic path ----
    let mut d_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();
    // SAFETY: launch shape matches the kernel; buffer covers its accesses.
    unsafe { module.exact_div_direct(&stream, cfg, &mut d_dev) }.expect("exact_div_direct launch");
    let dgot = d_dev.to_host_vec(&stream).unwrap();

    let mut dbad = 0;
    for (i, &g) in dgot.iter().enumerate().take(N) {
        let n = (i as u32) * 12;
        let want = (n / 4).wrapping_add(((n as i32) / 3) as u32);
        if g != want {
            if dbad < 5 {
                println!("  exact_div mismatch at {i}: got {g} want {want}");
            }
            dbad += 1;
        }
    }
    println!("exact_div direct: {} / {N} correct", N - dbad);

    if bad == 0 && dbad == 0 {
        println!("\nPASS");
    } else {
        println!("\nFAIL");
        std::process::exit(1);
    }
}
