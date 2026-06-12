/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#![allow(clippy::needless_range_loop)]

//! Smoke test for warp-shuffle intrinsics on pre-Volta arches (sm_6x).
//!
//! A single-warp reduction over [0,1,2,...,31] using shuffle_xor_f32 (a `shfl.sync.bfly.f32`).
//! Expected: every lane ends with the sum 0+1+...+31 = 496.0.
//!
//! Without `-mattr=+ptx60` on the `llc` invocation, this fails to compile on sm_61 with:
//!   LLVM ERROR: Cannot select: intrinsic %llvm.nvvm.shfl.sync.bfly.f32
//! because `shfl.sync.*` requires PTX ISA >= 6.0, and `llc`'s default PTX version for pre-Volta
//! targets is older. sm_70+ defaults to a PTX version >= 6.0, so it compiles there without the flag.
//!
//! Build and run with:
//!   cargo oxide run shfl_sync_smoke --arch sm_61

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread, warp};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// Single-warp sum via shuffle_xor butterfly. All 32 lanes end with the total.
    #[kernel]
    pub fn warp_sum(data: &[f32], mut out: DisjointSlice<f32>) {
        let gid = thread::index_1d();
        let lane = warp::lane_id();
        let mut val = data[gid.get()];
        val += warp::shuffle_xor_f32(val, 16);
        val += warp::shuffle_xor_f32(val, 8);
        val += warp::shuffle_xor_f32(val, 4);
        val += warp::shuffle_xor_f32(val, 2);
        val += warp::shuffle_xor_f32(val, 1);
        if lane == 0 {
            unsafe { *out.get_unchecked_mut(0) = val; }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let expected = 31.0 * 32.0 / 2.0; // 0+1+...+31 = 496

    let data_dev = DeviceBuffer::from_host(&stream, &data)?;
    let mut out_dev = DeviceBuffer::from_host(&stream, &vec![0.0f32; 1])?;

    let module = ctx.load_module_from_file("shfl_sync_smoke.ptx")?;
    let module = kernels::from_module(module)?;
    let cfg = LaunchConfig { grid_dim: (1, 1, 1), block_dim: (32, 1, 1), shared_mem_bytes: 0 };
    module.warp_sum(stream.as_ref(), cfg, &data_dev, &mut out_dev)?;
    stream.synchronize()?;
    let out = out_dev.to_host_vec(&stream)?;

    println!("  warp sum of [0..32) = {}  (expect {})", out[0], expected);
    if out[0] == expected {
        println!("  SUCCESS: shfl.sync.bfly lowered and executed correctly on this arch");
        Ok(())
    } else {
        println!("  FAILURE: got {}, expected {}", out[0], expected);
        std::process::exit(1)
    }
}
