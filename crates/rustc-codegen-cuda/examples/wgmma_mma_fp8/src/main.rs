/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runnable Hopper check for `wgmma.mma_async.m64n64k32.f32.e4m3.e4m3`.
//!
//! Build and run on H100/H200 with:
//!   cargo oxide run wgmma_mma_fp8 --arch sm_90a

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::barrier::fence_proxy_async_shared_cta;
use cuda_device::shared::SharedArray;
use cuda_device::swizzle::Swizzle;
use cuda_device::wgmma::{
    make_smem_desc, wgmma_commit_group, wgmma_fence, wgmma_mma_m64n64k16_f32_bf16,
    wgmma_mma_m64n64k32_f32_e4m3_e4m3, wgmma_wait_group,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;
use std::sync::Arc;

const M: usize = 64;
const N: usize = 64;
const FP8_K: usize = 32;
const BF16_K: usize = 16;
const TILE_BYTES: usize = 2048;
const SMEM_BYTES: usize = TILE_BYTES * 2;
const THREADS: usize = 128;
const ACCUM_VALUES: usize = 32;
const INITIAL_ACCUMULATOR: f32 = 1.0;

const BENCH_BLOCKS: u32 = 8192;
const BENCH_WARMUPS: usize = 10;
const BENCH_SAMPLES: usize = 11;
const BENCH_LAUNCHES: usize = 100;
const BENCH_EFFECTIVE_K: usize = 64;

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(always)]
    unsafe fn stage_tiles(smem: *mut u8, a: &[u8], b: &[u8]) {
        let tid = thread::threadIdx_x() as usize;
        let mut logical = tid;
        while logical < TILE_BYTES {
            let physical = Swizzle::<1, 4, 7>::apply(logical);
            unsafe {
                smem.add(physical).write(a[logical]);
                smem.add(TILE_BYTES + physical).write(b[logical]);
            }
            logical += THREADS;
        }

        // Each writer publishes its generic-proxy stores to the async proxy;
        // the CTA barrier then keeps WGMMA from racing any other writer.
        unsafe { fence_proxy_async_shared_cta() };
        thread::sync_threads();
    }

    #[kernel]
    pub fn fp8_correctness(a: &[u8], b: &[u8], mut out: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<u8, SMEM_BYTES, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SMEM) };
        unsafe { stage_tiles(smem, a, b) };

        let mut acc = [[INITIAL_ACCUMULATOR; 8]; 4];
        unsafe {
            let desc_a = make_smem_desc(&raw const SMEM as *const u8);
            let desc_b = make_smem_desc((&raw const SMEM as *const u8).add(TILE_BYTES));
            wgmma_fence();
            wgmma_mma_m64n64k32_f32_e4m3_e4m3(&mut acc, desc_a, desc_b);
            wgmma_commit_group();
            wgmma_wait_group::<0>();
        }

        let base = tid * ACCUM_VALUES;
        if base + ACCUM_VALUES <= out.len() {
            for (outer, row) in acc.iter().enumerate() {
                for (inner, &value) in row.iter().enumerate() {
                    let register = outer * 8 + inner;
                    // SAFETY: one 128-thread CTA owns 32 non-overlapping slots
                    // per thread, and the preceding check covers the full run.
                    unsafe { *out.get_unchecked_mut(base + register) = value };
                }
            }
        }
    }

    #[kernel]
    pub fn bf16_correctness(a: &[u8], b: &[u8], mut out: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<u8, SMEM_BYTES, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SMEM) };
        unsafe { stage_tiles(smem, a, b) };

        let mut acc = [[INITIAL_ACCUMULATOR; 8]; 4];
        unsafe {
            let desc_a = make_smem_desc(&raw const SMEM as *const u8);
            let desc_b = make_smem_desc((&raw const SMEM as *const u8).add(TILE_BYTES));
            wgmma_fence();
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            wgmma_commit_group();
            wgmma_wait_group::<0>();
        }

        let base = tid * ACCUM_VALUES;
        if base + ACCUM_VALUES <= out.len() {
            for (outer, row) in acc.iter().enumerate() {
                for (inner, &value) in row.iter().enumerate() {
                    let register = outer * 8 + inner;
                    // SAFETY: each thread writes its own 32-element fragment
                    // after the bounds check above.
                    unsafe { *out.get_unchecked_mut(base + register) = value };
                }
            }
        }
    }

    #[kernel]
    pub fn fp8_benchmark(a: &[u8], b: &[u8], mut checksum: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<u8, SMEM_BYTES, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SMEM) };
        unsafe { stage_tiles(smem, a, b) };

        let mut acc = [[INITIAL_ACCUMULATOR; 8]; 4];
        unsafe {
            let desc_a = make_smem_desc(&raw const SMEM as *const u8);
            let desc_b = make_smem_desc((&raw const SMEM as *const u8).add(TILE_BYTES));
            wgmma_fence();
            wgmma_mma_m64n64k32_f32_e4m3_e4m3(&mut acc, desc_a, desc_b);
            wgmma_mma_m64n64k32_f32_e4m3_e4m3(&mut acc, desc_a, desc_b);
            wgmma_commit_group();
            wgmma_wait_group::<0>();
        }

        if tid == 0 {
            let block = thread::blockIdx_x() as usize;
            if block < checksum.len() {
                // SAFETY: only thread 0 writes, and blockIdx.x is unique.
                unsafe { *checksum.get_unchecked_mut(block) = acc[0][0] };
            }
        }
    }

    #[kernel]
    pub fn bf16_benchmark(a: &[u8], b: &[u8], mut checksum: DisjointSlice<f32>) {
        static mut SMEM: SharedArray<u8, SMEM_BYTES, 256> = SharedArray::UNINIT;

        let tid = thread::threadIdx_x() as usize;
        let smem = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SMEM) };
        unsafe { stage_tiles(smem, a, b) };

        let mut acc = [[INITIAL_ACCUMULATOR; 8]; 4];
        unsafe {
            let desc_a = make_smem_desc(&raw const SMEM as *const u8);
            let desc_b = make_smem_desc((&raw const SMEM as *const u8).add(TILE_BYTES));
            wgmma_fence();
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            wgmma_mma_m64n64k16_f32_bf16(&mut acc, desc_a, desc_b);
            wgmma_commit_group();
            wgmma_wait_group::<0>();
        }

        if tid == 0 {
            let block = thread::blockIdx_x() as usize;
            if block < checksum.len() {
                // SAFETY: only thread 0 writes, and blockIdx.x is unique.
                unsafe { *checksum.get_unchecked_mut(block) = acc[0][0] };
            }
        }
    }
}

fn encode_e4m3(value: i32) -> u8 {
    let sign = if value < 0 { 0x80 } else { 0 };
    match value.abs() {
        0 => 0,
        1 => sign | 0x38,
        2 => sign | 0x40,
        3 => sign | 0x44,
        _ => panic!("encode_e4m3 only handles |value| <= 3"),
    }
}

fn decode_e4m3(bits: u8) -> f32 {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = ((bits >> 3) & 0x0f) as i32;
    let mantissa = (bits & 0x07) as u32;
    if exponent == 0x0f && mantissa == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        sign * mantissa as f32 * 2.0f32.powi(-9)
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent - 7)
    }
}

fn a_value(row: usize, k: usize) -> i32 {
    ((row + pattern_k(k)) % 7) as i32 - 3
}

fn b_value(k: usize, col: usize) -> i32 {
    ((pattern_k(k) * 2 + col) % 5) as i32 - 2
}

fn pattern_k(k: usize) -> usize {
    // Permute both operands' second K half together: the dot product stays
    // equal to two BF16 tiles, but a missing SW32 half-swap becomes visible.
    if k < BF16_K { k } else { (k + 1) % BF16_K }
}

struct Inputs {
    fp8_a: Vec<u8>,
    fp8_b: Vec<u8>,
    bf16_a: Vec<u8>,
    bf16_b: Vec<u8>,
}

fn make_inputs() -> Inputs {
    let mut fp8_a = vec![0u8; TILE_BYTES];
    let mut fp8_b = vec![0u8; TILE_BYTES];
    for row in 0..M {
        for k in 0..FP8_K {
            fp8_a[row * FP8_K + k] = encode_e4m3(a_value(row, k));
        }
    }
    for col in 0..N {
        for k in 0..FP8_K {
            fp8_b[col * FP8_K + k] = encode_e4m3(b_value(k, col));
        }
    }

    let mut bf16_a = vec![0u8; TILE_BYTES];
    let mut bf16_b = vec![0u8; TILE_BYTES];
    for row in 0..M {
        for k in 0..BF16_K {
            let bytes = (((a_value(row, k) as f32).to_bits() >> 16) as u16).to_le_bytes();
            let offset = 2 * (row * BF16_K + k);
            bf16_a[offset] = bytes[0];
            bf16_a[offset + 1] = bytes[1];
        }
    }
    for col in 0..N {
        for k in 0..BF16_K {
            let bytes = (((b_value(k, col) as f32).to_bits() >> 16) as u16).to_le_bytes();
            let offset = 2 * (col * BF16_K + k);
            bf16_b[offset] = bytes[0];
            bf16_b[offset + 1] = bytes[1];
        }
    }

    Inputs {
        fp8_a,
        fp8_b,
        bf16_a,
        bf16_b,
    }
}

fn fp8_at(bytes: &[u8], index: usize) -> f32 {
    decode_e4m3(bytes[index])
}

fn bf16_at(bytes: &[u8], index: usize) -> f32 {
    let offset = index * 2;
    f32::from_bits((u16::from_le_bytes([bytes[offset], bytes[offset + 1]]) as u32) << 16)
}

fn reference_matrix(
    a: &[u8],
    b: &[u8],
    k_extent: usize,
    repeats: usize,
    decode: fn(&[u8], usize) -> f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; M * N];
    for row in 0..M {
        for col in 0..N {
            let mut acc = INITIAL_ACCUMULATOR;
            for _ in 0..repeats {
                for k in 0..k_extent {
                    acc += decode(a, row * k_extent + k) * decode(b, col * k_extent + k);
                }
            }
            out[row * N + col] = acc;
        }
    }
    out
}

fn scatter_fragment(fragment: &[f32]) -> Vec<f32> {
    assert_eq!(fragment.len(), THREADS * ACCUM_VALUES);
    let mut matrix = vec![f32::NAN; M * N];
    let mut seen = vec![false; M * N];
    for tid in 0..THREADS {
        let warp = tid / 32;
        let lane = tid % 32;
        for register in 0..ACCUM_VALUES {
            let row = lane / 4 + 16 * warp + 8 * ((register / 2) % 2);
            let col = 2 * (lane % 4) + register % 2 + 8 * (register / 4);
            let matrix_index = row * N + col;
            assert!(!seen[matrix_index], "duplicate accumulator mapping");
            seen[matrix_index] = true;
            matrix[matrix_index] = fragment[tid * ACCUM_VALUES + register];
        }
    }
    assert!(seen.into_iter().all(|value| value));
    matrix
}

fn check_values(
    label: &str,
    actual: &[f32],
    expected: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    if actual.len() != expected.len() {
        return Err(format!("{label}: length {} != {}", actual.len(), expected.len()).into());
    }
    let mut mismatches = 0usize;
    let mut first = None;
    let mut max_abs_error = 0.0f32;
    for (index, (&got, &want)) in actual.iter().zip(expected).enumerate() {
        if got != want {
            mismatches += 1;
            first.get_or_insert((index, got, want));
            let error = (got - want).abs();
            max_abs_error = if error.is_finite() {
                max_abs_error.max(error)
            } else {
                f32::INFINITY
            };
        }
    }
    if let Some((index, got, want)) = first {
        return Err(format!(
            "{label}: {mismatches} mismatches; max abs error: {max_abs_error}; first at {index}: {got} != {want}"
        )
        .into());
    }
    println!("{label}: checked {} values; max abs error: 0", actual.len());
    Ok(())
}

fn verify_ptx_only() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wgmma_mma_fp8.ptx");
    let ptx = std::fs::read_to_string(&path)?;
    for required in [
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3",
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16",
        "fence.proxy.async.shared::cta",
        "wgmma.fence.sync.aligned",
        "wgmma.commit_group.sync.aligned",
        "wgmma.wait_group.sync.aligned 0",
    ] {
        if !ptx.contains(required) {
            return Err(format!("generated PTX lacks `{required}`").into());
        }
    }
    println!(
        "WARNING: WGMMA requires Hopper; generated PTX contains the required FP8 and BF16 paths: {}",
        path.display()
    );
    Ok(())
}

fn run_correctness(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    fp8_a: &DeviceBuffer<u8>,
    fp8_b: &DeviceBuffer<u8>,
    bf16_a: &DeviceBuffer<u8>,
    bf16_b: &DeviceBuffer<u8>,
    inputs: &Inputs,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut fp8_out = DeviceBuffer::<f32>::zeroed(stream, M * N)?;
    let mut bf16_out = DeviceBuffer::<f32>::zeroed(stream, M * N)?;

    unsafe { module.fp8_correctness(stream.as_ref(), cfg, fp8_a, fp8_b, &mut fp8_out) }?;
    unsafe { module.bf16_correctness(stream.as_ref(), cfg, bf16_a, bf16_b, &mut bf16_out) }?;

    let fp8_matrix = scatter_fragment(&fp8_out.to_host_vec(stream)?);
    let bf16_matrix = scatter_fragment(&bf16_out.to_host_vec(stream)?);
    let fp8_reference = reference_matrix(&inputs.fp8_a, &inputs.fp8_b, FP8_K, 1, fp8_at);
    let bf16_reference = reference_matrix(&inputs.bf16_a, &inputs.bf16_b, BF16_K, 1, bf16_at);
    check_values("FP8 m64n64k32 numeric check", &fp8_matrix, &fp8_reference)?;
    check_values(
        "BF16 m64n64k16 baseline check",
        &bf16_matrix,
        &bf16_reference,
    )?;
    Ok(())
}

fn measure_fp8(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    cfg: LaunchConfig,
    a: &DeviceBuffer<u8>,
    b: &DeviceBuffer<u8>,
    checksum: &mut DeviceBuffer<f32>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..BENCH_LAUNCHES {
        unsafe { module.fp8_benchmark(stream.as_ref(), cfg, a, b, &mut *checksum) }?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    Ok(start.elapsed_ms(&end)? as f64 / BENCH_LAUNCHES as f64)
}

fn measure_bf16(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    cfg: LaunchConfig,
    a: &DeviceBuffer<u8>,
    b: &DeviceBuffer<u8>,
    checksum: &mut DeviceBuffer<f32>,
) -> Result<f64, Box<dyn std::error::Error>> {
    let start = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    for _ in 0..BENCH_LAUNCHES {
        unsafe { module.bf16_benchmark(stream.as_ref(), cfg, a, b, &mut *checksum) }?;
    }
    let end = stream.record_event(Some(cuda_core::sys::CUevent_flags_enum_CU_EVENT_DEFAULT))?;
    Ok(start.elapsed_ms(&end)? as f64 / BENCH_LAUNCHES as f64)
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[allow(clippy::too_many_arguments)]
fn run_benchmark(
    stream: &Arc<CudaStream>,
    module: &kernels::LoadedModule,
    fp8_a: &DeviceBuffer<u8>,
    fp8_b: &DeviceBuffer<u8>,
    bf16_a: &DeviceBuffer<u8>,
    bf16_b: &DeviceBuffer<u8>,
    inputs: &Inputs,
    check_only: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let cfg = LaunchConfig {
        grid_dim: (BENCH_BLOCKS, 1, 1),
        block_dim: (THREADS as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut fp8_checksum = DeviceBuffer::<f32>::zeroed(stream, BENCH_BLOCKS as usize)?;
    let mut bf16_checksum = DeviceBuffer::<f32>::zeroed(stream, BENCH_BLOCKS as usize)?;

    let fp8_reference = reference_matrix(&inputs.fp8_a, &inputs.fp8_b, FP8_K, 2, fp8_at);
    let bf16_reference = reference_matrix(&inputs.bf16_a, &inputs.bf16_b, BF16_K, 4, bf16_at);
    check_values(
        "effective-K=64 FP8/BF16 reference agreement",
        &fp8_reference,
        &bf16_reference,
    )?;

    unsafe { module.fp8_benchmark(stream.as_ref(), cfg, fp8_a, fp8_b, &mut fp8_checksum) }?;
    unsafe { module.bf16_benchmark(stream.as_ref(), cfg, bf16_a, bf16_b, &mut bf16_checksum) }?;
    let expected_checksum = vec![fp8_reference[0]; BENCH_BLOCKS as usize];
    check_values(
        "FP8 benchmark checksum",
        &fp8_checksum.to_host_vec(stream)?,
        &expected_checksum,
    )?;
    check_values(
        "BF16 benchmark checksum",
        &bf16_checksum.to_host_vec(stream)?,
        &expected_checksum,
    )?;

    // Exercise both chained-MMA kernels under sanitizer without timing them.
    if check_only {
        return Ok(());
    }

    println!(
        "Tile microbenchmark: {M}x{N}x{BENCH_EFFECTIVE_K}, {BENCH_BLOCKS} CTAs, \
         {THREADS} threads/CTA, {SMEM_BYTES} shared bytes/CTA; \
         {BENCH_WARMUPS} warmups, {BENCH_SAMPLES} samples, {BENCH_LAUNCHES} launches/sample"
    );
    for warmup in 0..BENCH_WARMUPS {
        if warmup % 2 == 0 {
            unsafe { module.fp8_benchmark(stream.as_ref(), cfg, fp8_a, fp8_b, &mut fp8_checksum) }?;
            unsafe {
                module.bf16_benchmark(stream.as_ref(), cfg, bf16_a, bf16_b, &mut bf16_checksum)
            }?;
        } else {
            unsafe {
                module.bf16_benchmark(stream.as_ref(), cfg, bf16_a, bf16_b, &mut bf16_checksum)
            }?;
            unsafe { module.fp8_benchmark(stream.as_ref(), cfg, fp8_a, fp8_b, &mut fp8_checksum) }?;
        }
    }
    stream.synchronize()?;

    let mut fp8_ms = Vec::with_capacity(BENCH_SAMPLES);
    let mut bf16_ms = Vec::with_capacity(BENCH_SAMPLES);
    for sample in 0..BENCH_SAMPLES {
        if sample % 2 == 0 {
            fp8_ms.push(measure_fp8(
                stream,
                module,
                cfg,
                fp8_a,
                fp8_b,
                &mut fp8_checksum,
            )?);
            bf16_ms.push(measure_bf16(
                stream,
                module,
                cfg,
                bf16_a,
                bf16_b,
                &mut bf16_checksum,
            )?);
        } else {
            bf16_ms.push(measure_bf16(
                stream,
                module,
                cfg,
                bf16_a,
                bf16_b,
                &mut bf16_checksum,
            )?);
            fp8_ms.push(measure_fp8(
                stream,
                module,
                cfg,
                fp8_a,
                fp8_b,
                &mut fp8_checksum,
            )?);
        }
    }

    if fp8_ms
        .iter()
        .chain(&bf16_ms)
        .any(|ms| !ms.is_finite() || *ms <= 0.0)
    {
        return Err("CUDA event timing must be finite and positive".into());
    }
    println!("FP8 sample mean launch times (ms): {fp8_ms:.6?}");
    println!("BF16 sample mean launch times (ms): {bf16_ms:.6?}");
    let fp8_median_ms = median(fp8_ms);
    let bf16_median_ms = median(bf16_ms);
    let flops = BENCH_BLOCKS as f64 * 2.0 * M as f64 * N as f64 * BENCH_EFFECTIVE_K as f64;
    let fp8_tflops = flops / (fp8_median_ms / 1000.0) / 1.0e12;
    let bf16_tflops = flops / (bf16_median_ms / 1000.0) / 1.0e12;
    println!("BF16 m64n64k16 x4: {bf16_median_ms:.4} ms, {bf16_tflops:.2} TFLOPS");
    println!("FP8  m64n64k32 x2: {fp8_median_ms:.4} ms, {fp8_tflops:.2} TFLOPS");
    check_values(
        "FP8 post-timing checksum",
        &fp8_checksum.to_host_vec(stream)?,
        &expected_checksum,
    )?;
    check_values(
        "BF16 post-timing checksum",
        &bf16_checksum.to_host_vec(stream)?,
        &expected_checksum,
    )?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let check_only = match args.as_slice() {
        [] => false,
        [arg] if arg == "--check-only" => true,
        _ => return Err("usage: wgmma_mma_fp8 [--check-only]".into()),
    };
    println!("=== FP8 WGMMA e4m3 x e4m3 -> f32 ===");
    let context = CudaContext::new(0)?;
    println!("GPU: {}", context.device_name()?);
    let (major, minor) = context.compute_capability()?;
    println!("GPU Compute Capability: sm_{major}{minor}");
    if major != 9 {
        if check_only {
            return Err("runtime validation requires an H100/H200 (sm_90a)".into());
        }
        return verify_ptx_only();
    }

    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let inputs = make_inputs();
    let fp8_a = DeviceBuffer::from_host(&stream, &inputs.fp8_a)?;
    let fp8_b = DeviceBuffer::from_host(&stream, &inputs.fp8_b)?;
    let bf16_a = DeviceBuffer::from_host(&stream, &inputs.bf16_a)?;
    let bf16_b = DeviceBuffer::from_host(&stream, &inputs.bf16_b)?;

    run_correctness(&stream, &module, &fp8_a, &fp8_b, &bf16_a, &bf16_b, &inputs)?;
    run_benchmark(
        &stream, &module, &fp8_a, &fp8_b, &bf16_a, &bf16_b, &inputs, check_only,
    )?;
    println!("SUCCESS: FP8 WGMMA numeric check and BF16 comparison passed");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_check_reports_the_largest_error_and_rejects_nonfinite_results() {
        let error = check_values("finite", &[2.0, -4.0], &[1.0, 2.0])
            .unwrap_err()
            .to_string();
        assert!(error.contains("2 mismatches"));
        assert!(error.contains("max abs error: 6"));
        assert!(error.contains("first at 0: 2 != 1"));

        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let error = check_values("nonfinite", &[value], &[1.0])
                .unwrap_err()
                .to_string();
            assert!(error.contains("1 mismatches"));
            assert!(error.contains("max abs error: inf"));
        }
    }

    #[test]
    fn small_integer_e4m3_roundtrip_is_exact() {
        for value in -3..=3 {
            assert_eq!(decode_e4m3(encode_e4m3(value)), value as f32);
        }
    }

    #[test]
    fn accumulator_scatter_is_bijective() {
        let fragment: Vec<f32> = (0..M * N).map(|index| index as f32).collect();
        let mut matrix = scatter_fragment(&fragment);
        matrix.sort_by(f32::total_cmp);
        assert_eq!(matrix, fragment);
    }

    #[test]
    fn benchmark_references_match_and_exercise_the_product() {
        let inputs = make_inputs();
        let fp8 = reference_matrix(&inputs.fp8_a, &inputs.fp8_b, FP8_K, 2, fp8_at);
        let bf16 = reference_matrix(&inputs.bf16_a, &inputs.bf16_b, BF16_K, 4, bf16_at);
        assert_ne!(&inputs.fp8_a[..BF16_K], &inputs.fp8_a[BF16_K..FP8_K]);
        assert_ne!(&inputs.fp8_b[..BF16_K], &inputs.fp8_b[BF16_K..FP8_K]);
        assert_eq!(fp8, bf16);
        assert!(fp8.iter().any(|&value| value != INITIAL_ACCUMULATOR));
        assert!(fp8.iter().any(|&value| value != 0.0));
    }
}
