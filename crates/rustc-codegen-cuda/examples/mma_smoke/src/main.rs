/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Smoke test for two SM80+ tensor-core intrinsics:
//!
//! 1. `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` (`mma_smoke_kernel`)
//! 2. `ldmatrix.sync.aligned.m8n8.x4.shared.b16` and the `.trans` variant
//!    (`ldmatrix_smoke_kernel`)
//!
//! Each kernel launches one warp (32 threads, 1 block). Per-lane register
//! fragments are written to a contiguous output buffer (4 values per lane).
//!
//! Correctness validation is intentionally minimal -- the goal is to confirm
//! that the PTX lines are emitted and the kernels run without crashing.

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::ldmatrix::{ldmatrix_x4_b16, ldmatrix_x4_trans_b16};
use cuda_device::mma::mma_m16n8k16_bf16_f32;
use cuda_device::shared::SharedArray;
use cuda_device::{DisjointSlice, kernel, sync_threads, thread};
use cuda_host::cuda_launch;
use std::sync::Arc;

#[kernel]
pub unsafe fn mma_smoke_kernel(mut output: DisjointSlice<f32>) {
    let tid = thread::threadIdx_x();

    // Constant register inputs. These won't produce anything mathematically
    // meaningful -- the point is to exercise the emit path.
    let a0: u32 = 0x3F803F80; // bf16(1.0) | bf16(1.0) packed
    let a1: u32 = 0x3F803F80;
    let a2: u32 = 0x3F803F80;
    let a3: u32 = 0x3F803F80;
    let b0: u32 = 0x3F803F80;
    let b1: u32 = 0x3F803F80;
    let c0: f32 = 0.0;
    let c1: f32 = 0.0;
    let c2: f32 = 0.0;
    let c3: f32 = 0.0;

    let d = unsafe { mma_m16n8k16_bf16_f32(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3) };

    let base = (tid as usize) * 4;
    let len = output.len();
    let ptr = output.as_mut_ptr();
    unsafe {
        if base + 3 < len {
            *ptr.add(base) = d.x();
            *ptr.add(base + 1) = d.y();
            *ptr.add(base + 2) = d.z();
            *ptr.add(base + 3) = d.w();
        }
    }
}

/// 16x16 b16 tile: 256 elements stored as 128 packed bf16x2 u32 words.
/// 4 8x8 tiles of b16 = 4 * 64 = 256 elements = 128 u32s.
const SMEM_U32_LEN: usize = 128;

#[kernel]
pub unsafe fn ldmatrix_smoke_kernel(
    mut plain_out: DisjointSlice<u32>,
    mut trans_out: DisjointSlice<u32>,
) {
    // Shared scratch for the four 8x8 b16 tiles.
    static mut SMEM: SharedArray<u32, SMEM_U32_LEN> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x();

    // Cooperatively initialize SMEM with a recognizable pattern: each u32
    // word i = (i << 16) | i (so low/high b16 halves both equal i).
    unsafe {
        let smem_ptr = (&raw mut SMEM).cast::<SharedArray<u32, SMEM_U32_LEN>>();
        let base = (*smem_ptr).as_mut_ptr();
        let i = tid as usize;
        if i < SMEM_U32_LEN {
            let v: u32 = ((i as u32) << 16) | (i as u32);
            *base.add(i) = v;
        }
        // Second half (32..128) -- thread 0 fills it to keep things simple.
        if tid == 0 {
            let mut j = 32usize;
            while j < SMEM_U32_LEN {
                let v: u32 = ((j as u32) << 16) | (j as u32);
                *base.add(j) = v;
                j += 1;
            }
        }
    }
    sync_threads();

    // Per-lane row pointer: lane k provides &SMEM[k * 4] -- ldmatrix.x4 uses
    // lanes 0..31 to address 32 8-byte rows total (8 rows x 4 matrices).
    let row_off = (tid as usize) * 4;
    let smem_base_ptr: *const u8 = unsafe {
        let smem_ptr = (&raw mut SMEM).cast::<SharedArray<u32, SMEM_U32_LEN>>();
        (*smem_ptr).as_ptr().add(row_off).cast::<u8>()
    };

    let plain = unsafe { ldmatrix_x4_b16(smem_base_ptr) };
    let trans = unsafe { ldmatrix_x4_trans_b16(smem_base_ptr) };

    let out_base = (tid as usize) * 4;

    let plain_len = plain_out.len();
    let plain_ptr = plain_out.as_mut_ptr();
    unsafe {
        if out_base + 3 < plain_len {
            *plain_ptr.add(out_base) = plain.x();
            *plain_ptr.add(out_base + 1) = plain.y();
            *plain_ptr.add(out_base + 2) = plain.z();
            *plain_ptr.add(out_base + 3) = plain.w();
        }
    }

    let trans_len = trans_out.len();
    let trans_ptr = trans_out.as_mut_ptr();
    unsafe {
        if out_base + 3 < trans_len {
            *trans_ptr.add(out_base) = trans.x();
            *trans_ptr.add(out_base + 1) = trans.y();
            *trans_ptr.add(out_base + 2) = trans.z();
            *trans_ptr.add(out_base + 3) = trans.w();
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mma.sync m16n8k16 + ldmatrix.x4 smoke test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("mma_smoke.ptx");
    println!("Loading PTX from: {}", ptx_path.display());
    let ptx_file = ptx_path.to_str().ok_or("PTX path is not valid UTF-8")?;
    let module = ctx.load_module_from_file(ptx_file)?;
    println!("PTX loaded\n");

    run_mma_smoke(&stream, &module)?;
    run_ldmatrix_smoke(&stream, &module)?;
    Ok(())
}

fn run_mma_smoke(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut dev_output = DeviceBuffer::<f32>::zeroed(stream, 128)?;

    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("Launching mma_smoke_kernel (32 threads, 1 block)...");
    cuda_launch! {
        kernel: mma_smoke_kernel,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(dev_output)]
    }?;
    stream.synchronize()?;

    let host_output = dev_output.to_host_vec(stream)?;
    println!("First 8 D-fragment values: {:?}", &host_output[..8]);
    Ok(())
}

fn run_ldmatrix_smoke(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut dev_plain = DeviceBuffer::<u32>::zeroed(stream, 128)?;
    let mut dev_trans = DeviceBuffer::<u32>::zeroed(stream, 128)?;

    let cfg = LaunchConfig {
        block_dim: (32, 1, 1),
        grid_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    println!("\nLaunching ldmatrix_smoke_kernel (32 threads, 1 block)...");
    cuda_launch! {
        kernel: ldmatrix_smoke_kernel,
        stream: stream,
        module: module,
        config: cfg,
        args: [slice_mut(dev_plain), slice_mut(dev_trans)]
    }?;
    stream.synchronize()?;

    let plain = dev_plain.to_host_vec(stream)?;
    let trans = dev_trans.to_host_vec(stream)?;
    println!("Lane 0 plain frag : {:08x?}", &plain[0..4]);
    println!("Lane 0 trans frag : {:08x?}", &trans[0..4]);
    println!("Lane 1 plain frag : {:08x?}", &plain[4..8]);
    Ok(())
}
