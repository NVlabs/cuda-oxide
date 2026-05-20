/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! bf16 -> f32 GEMM using mma.sync.aligned.m16n8k16 + ldmatrix.x4 (SM80+).
//!
//! 4-warp 64x64 block tile. K=16 per step. Each warp owns a 32x32 chunk of
//! the output and issues 8 mma.m16n8k16 instructions per K-step
//! (m_tile in {0,1} x n_tile in {0,1,2,3}).
//!
//! Build / run:
//!   cargo oxide build --arch sm_120 mma_gemm
//!   cargo oxide run   --arch sm_120 mma_gemm verify
//!   cargo oxide run   --arch sm_120 mma_gemm bench

use cuda_core::{CudaContext, CudaModule, CudaStream, DeviceBuffer, LaunchConfig};
use cuda_device::ldmatrix::{ldmatrix_x4_b16, ldmatrix_x4_trans_b16};
use cuda_device::mma::mma_m16n8k16_bf16_f32;
use cuda_device::shared::SharedArray;
use cuda_device::{DisjointSlice, kernel, sync_threads, thread};
use cuda_host::cuda_launch;
use half::bf16;
use std::sync::Arc;
use std::time::Instant;

// =============================================================================
// KERNEL
// =============================================================================
//
// Block tile: BM=64 rows x BN=64 cols of C, BK=16 per K-step.
//
// SMEM_A: 64 x 16 bf16, row-major. Row stride = 16 bf16 = 32 bytes = 8 u32.
//         Total = 64*8 = 512 u32 = 2048 bytes.
// SMEM_B: 16 x 64 bf16, row-major B[k, n]. Row stride = 64 bf16 = 128 bytes = 32 u32.
//         Total = 16*32 = 512 u32 = 2048 bytes.
//
// 128 threads/block (4 warps). Each thread cooperatively loads 4 u32 of A
// and 4 u32 of B per K-step (128*4 = 512 u32 each).
//
// Warp layout (warp_id = tid / 32):
//   warp 0 -> rows [0, 32), cols [0, 32)
//   warp 1 -> rows [0, 32), cols [32, 64)
//   warp 2 -> rows [32, 64), cols [0, 32)
//   warp 3 -> rows [32, 64), cols [32, 64)
//
// Per warp per K=16 step:
//   2 ldmatrix.x4 (A top 16x16, A bot 16x16)
//   2 ldmatrix.x4.trans (B left 16x16, B right 16x16)
//   8 mma.m16n8k16 -> acc[2][4][4 f32]
const SMEM_A_LEN_U32: usize = 64 * 8; // 512
const SMEM_B_LEN_U32: usize = 16 * 32; // 512

#[kernel]
pub unsafe fn mma_gemm_kernel(
    m: u32,
    n: u32,
    k: u32,
    a: &[u16], // bf16-as-u16, row-major MxK
    b: &[u16], // bf16-as-u16, row-major KxN
    mut c: DisjointSlice<f32>, // f32, row-major MxN
) {
    static mut SMEM_A: SharedArray<u32, SMEM_A_LEN_U32> = SharedArray::UNINIT;
    static mut SMEM_B: SharedArray<u32, SMEM_B_LEN_U32> = SharedArray::UNINIT;

    let tid = thread::threadIdx_x() as usize;
    let warp_id = tid >> 5; // 0..4
    let lane = tid & 31;

    // Warp's 32x32 chunk origin within the block tile.
    let warp_row = (warp_id >> 1) * 32; // {0, 32}
    let warp_col = (warp_id & 1) * 32; // {0, 32}

    let block_m = thread::blockIdx_y() as usize * 64;
    let block_n = thread::blockIdx_x() as usize * 64;

    let m_size = m as usize;
    let n_size = n as usize;
    let k_size = k as usize;

    // 8 mma fragments per warp, 4 f32 each -> 32 f32 accumulators per lane.
    // Hand-unrolled scalar accumulators: acc{m}{n}_{0..3} where m=top|bot, n=0..3.
    let mut at_n0_0: f32 = 0.0; let mut at_n0_1: f32 = 0.0; let mut at_n0_2: f32 = 0.0; let mut at_n0_3: f32 = 0.0;
    let mut at_n1_0: f32 = 0.0; let mut at_n1_1: f32 = 0.0; let mut at_n1_2: f32 = 0.0; let mut at_n1_3: f32 = 0.0;
    let mut at_n2_0: f32 = 0.0; let mut at_n2_1: f32 = 0.0; let mut at_n2_2: f32 = 0.0; let mut at_n2_3: f32 = 0.0;
    let mut at_n3_0: f32 = 0.0; let mut at_n3_1: f32 = 0.0; let mut at_n3_2: f32 = 0.0; let mut at_n3_3: f32 = 0.0;
    let mut ab_n0_0: f32 = 0.0; let mut ab_n0_1: f32 = 0.0; let mut ab_n0_2: f32 = 0.0; let mut ab_n0_3: f32 = 0.0;
    let mut ab_n1_0: f32 = 0.0; let mut ab_n1_1: f32 = 0.0; let mut ab_n1_2: f32 = 0.0; let mut ab_n1_3: f32 = 0.0;
    let mut ab_n2_0: f32 = 0.0; let mut ab_n2_1: f32 = 0.0; let mut ab_n2_2: f32 = 0.0; let mut ab_n2_3: f32 = 0.0;
    let mut ab_n3_0: f32 = 0.0; let mut ab_n3_1: f32 = 0.0; let mut ab_n3_2: f32 = 0.0; let mut ab_n3_3: f32 = 0.0;

    let num_ktiles = k_size / 16;
    let mut kt = 0usize;
    while kt < num_ktiles {
        let k_start = kt * 16;

        // ---------- Cooperative load A tile: 64 rows x 16 cols bf16 ----------
        // 128 threads, 4 u32 each = 512 u32 total. SMEM_A indexed
        // u32[row*8 + col_word]; 8 u32 per row of 16 bf16.
        unsafe {
            let smem_a = (&raw mut SMEM_A)
                .cast::<SharedArray<u32, SMEM_A_LEN_U32>>()
                .as_mut()
                .unwrap()
                .as_mut_ptr();
            let mut i = 0usize;
            while i < 4 {
                let word_idx = tid * 4 + i;
                let row = word_idx / 8; // 0..64
                let col_word = word_idx % 8; // 0..8
                let col_lo = col_word * 2;
                let g_row = block_m + row;
                let g_col_lo = k_start + col_lo;
                // Hot path: when both halves are in-bounds, do a single u32 load.
                let packed: u32 = if g_row < m_size && g_col_lo + 2 <= k_size {
                    let p = a.as_ptr().add(g_row * k_size + g_col_lo) as *const u32;
                    *p
                } else if g_row < m_size && g_col_lo < k_size {
                    let lo = *a.as_ptr().add(g_row * k_size + g_col_lo) as u32;
                    lo
                } else {
                    0
                };
                *smem_a.add(word_idx) = packed;
                i += 1;
            }
        }

        // ---------- Cooperative load B tile: 16 rows x 64 cols bf16 ----------
        // Row-major B[k, n]; ldmatrix.x4.trans transposes during load.
        // 512 u32 total, 4 per thread. SMEM_B indexed u32[row*32 + col_word];
        // 32 u32 per row of 64 bf16.
        unsafe {
            let smem_b = (&raw mut SMEM_B)
                .cast::<SharedArray<u32, SMEM_B_LEN_U32>>()
                .as_mut()
                .unwrap()
                .as_mut_ptr();
            let mut i = 0usize;
            while i < 4 {
                let word_idx = tid * 4 + i;
                let row = word_idx / 32; // 0..16
                let col_word = word_idx % 32; // 0..32
                let col_lo = col_word * 2;
                let g_row = k_start + row;
                let g_col_lo = block_n + col_lo;
                let packed: u32 = if g_row < k_size && g_col_lo + 2 <= n_size {
                    let p = b.as_ptr().add(g_row * n_size + g_col_lo) as *const u32;
                    *p
                } else if g_row < k_size && g_col_lo < n_size {
                    let lo = *b.as_ptr().add(g_row * n_size + g_col_lo) as u32;
                    lo
                } else {
                    0
                };
                *smem_b.add(word_idx) = packed;
                i += 1;
            }
        }

        sync_threads();

        // ---------- ldmatrix on A: 2 calls per warp ----------
        // Each ldmatrix.x4 lane t addresses: row = (t & 15), col_byte = ((t>>4)&1)*16
        // within a 16x16 sub-tile (stride = 32 bytes = SMEM_A row stride).
        let a_inner_row = lane & 15;
        let a_col_byte = ((lane >> 4) & 1) * 16;

        // A_top covers warp rows [warp_row, warp_row+16); A_bot covers [warp_row+16, warp_row+32).
        let a_top = unsafe {
            let smem_a_base = (&raw const SMEM_A).cast::<u8>();
            let lane_ptr = smem_a_base.add((warp_row + a_inner_row) * 32 + a_col_byte);
            ldmatrix_x4_b16(lane_ptr)
        };
        let a_bot = unsafe {
            let smem_a_base = (&raw const SMEM_A).cast::<u8>();
            let lane_ptr = smem_a_base.add((warp_row + 16 + a_inner_row) * 32 + a_col_byte);
            ldmatrix_x4_b16(lane_ptr)
        };

        // ---------- ldmatrix.trans on B: 2 calls per warp ----------
        // SMEM_B row stride = 128 bytes. Lane t addresses: row=(t&15),
        // col_byte = warp_col_byte + ((t>>4)&1)*16. Each ldmatrix.x4.trans
        // covers a 16x16 col-region of the K=16 slab, producing 4 b32/lane,
        // i.e. TWO mma N=8 fragments worth (low pair = first 8 cols, high pair = next 8).
        let b_inner_row = lane & 15;
        let b_extra_col_byte = ((lane >> 4) & 1) * 16;
        let warp_col_byte = warp_col * 2; // bf16 = 2 bytes
        let b_left = unsafe {
            let smem_b_base = (&raw const SMEM_B).cast::<u8>();
            let lane_ptr = smem_b_base.add(b_inner_row * 128 + warp_col_byte + b_extra_col_byte);
            ldmatrix_x4_trans_b16(lane_ptr)
        };
        let b_right = unsafe {
            let smem_b_base = (&raw const SMEM_B).cast::<u8>();
            let lane_ptr = smem_b_base.add(b_inner_row * 128 + (warp_col_byte + 32) + b_extra_col_byte);
            ldmatrix_x4_trans_b16(lane_ptr)
        };

        // ---------- 8 mma.m16n8k16 per warp ----------
        // m_tile selects A_top vs A_bot. n_tile selects which 2 b32s of which B fragment.
        //   n_tile 0 -> b_left.x, b_left.y
        //   n_tile 1 -> b_left.z, b_left.w
        //   n_tile 2 -> b_right.x, b_right.y
        //   n_tile 3 -> b_right.z, b_right.w
        let bx0 = b_left.x();
        let bx1 = b_left.y();
        let bx2 = b_left.z();
        let bx3 = b_left.w();
        let bx4 = b_right.x();
        let bx5 = b_right.y();
        let bx6 = b_right.z();
        let bx7 = b_right.w();

        let at0 = a_top.x();
        let at1 = a_top.y();
        let at2 = a_top.z();
        let at3 = a_top.w();
        let ab0 = a_bot.x();
        let ab1 = a_bot.y();
        let ab2 = a_bot.z();
        let ab3 = a_bot.w();

        // 8 mma calls (m_tile in {top,bot} x n_tile in 0..4), fully unrolled.
        unsafe {
            let d = mma_m16n8k16_bf16_f32(at0, at1, at2, at3, bx0, bx1, at_n0_0, at_n0_1, at_n0_2, at_n0_3);
            at_n0_0 = d.x(); at_n0_1 = d.y(); at_n0_2 = d.z(); at_n0_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(at0, at1, at2, at3, bx2, bx3, at_n1_0, at_n1_1, at_n1_2, at_n1_3);
            at_n1_0 = d.x(); at_n1_1 = d.y(); at_n1_2 = d.z(); at_n1_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(at0, at1, at2, at3, bx4, bx5, at_n2_0, at_n2_1, at_n2_2, at_n2_3);
            at_n2_0 = d.x(); at_n2_1 = d.y(); at_n2_2 = d.z(); at_n2_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(at0, at1, at2, at3, bx6, bx7, at_n3_0, at_n3_1, at_n3_2, at_n3_3);
            at_n3_0 = d.x(); at_n3_1 = d.y(); at_n3_2 = d.z(); at_n3_3 = d.w();

            let d = mma_m16n8k16_bf16_f32(ab0, ab1, ab2, ab3, bx0, bx1, ab_n0_0, ab_n0_1, ab_n0_2, ab_n0_3);
            ab_n0_0 = d.x(); ab_n0_1 = d.y(); ab_n0_2 = d.z(); ab_n0_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(ab0, ab1, ab2, ab3, bx2, bx3, ab_n1_0, ab_n1_1, ab_n1_2, ab_n1_3);
            ab_n1_0 = d.x(); ab_n1_1 = d.y(); ab_n1_2 = d.z(); ab_n1_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(ab0, ab1, ab2, ab3, bx4, bx5, ab_n2_0, ab_n2_1, ab_n2_2, ab_n2_3);
            ab_n2_0 = d.x(); ab_n2_1 = d.y(); ab_n2_2 = d.z(); ab_n2_3 = d.w();
            let d = mma_m16n8k16_bf16_f32(ab0, ab1, ab2, ab3, bx6, bx7, ab_n3_0, ab_n3_1, ab_n3_2, ab_n3_3);
            ab_n3_0 = d.x(); ab_n3_1 = d.y(); ab_n3_2 = d.z(); ab_n3_3 = d.w();
        }

        sync_threads();
        kt += 1;
    }

    // ---------- Epilogue: per-lane mma D layout, store to global C ----------
    // For each (m_tile, n_tile) within the warp's 32x32 region:
    //   block-relative row base = warp_row + m_tile*16
    //   block-relative col base = warp_col + n_tile*8
    //   group = lane / 4, tid_g = lane % 4
    //   d[0] -> (row=group,    col=tid_g*2)
    //   d[1] -> (row=group,    col=tid_g*2+1)
    //   d[2] -> (row=group+8,  col=tid_g*2)
    //   d[3] -> (row=group+8,  col=tid_g*2+1)
    let group = lane / 4;
    let tid_g = lane % 4;

    unsafe {
        let c_len = c.len();
        let c_ptr = c.as_mut_ptr();

        // Helper closure-equivalent inlined: store one mma D-fragment.
        // base_row = block_m + warp_row + m_tile*16; base_col = block_n + warp_col + n_tile*8.
        macro_rules! store_frag {
            ($base_row:expr, $base_col:expr, $v0:expr, $v1:expr, $v2:expr, $v3:expr) => {{
                let r_top = $base_row + group;
                let r_bot = $base_row + group + 8;
                let c0 = $base_col + tid_g * 2;
                let c0p = c0 + 1;
                if r_top < m_size && c0 < n_size {
                    let idx = r_top * n_size + c0;
                    if idx < c_len { *c_ptr.add(idx) = $v0; }
                }
                if r_top < m_size && c0p < n_size {
                    let idx = r_top * n_size + c0p;
                    if idx < c_len { *c_ptr.add(idx) = $v1; }
                }
                if r_bot < m_size && c0 < n_size {
                    let idx = r_bot * n_size + c0;
                    if idx < c_len { *c_ptr.add(idx) = $v2; }
                }
                if r_bot < m_size && c0p < n_size {
                    let idx = r_bot * n_size + c0p;
                    if idx < c_len { *c_ptr.add(idx) = $v3; }
                }
            }};
        }

        let row_top = block_m + warp_row;
        let row_bot = block_m + warp_row + 16;
        let col0 = block_n + warp_col;
        let col1 = block_n + warp_col + 8;
        let col2 = block_n + warp_col + 16;
        let col3 = block_n + warp_col + 24;

        store_frag!(row_top, col0, at_n0_0, at_n0_1, at_n0_2, at_n0_3);
        store_frag!(row_top, col1, at_n1_0, at_n1_1, at_n1_2, at_n1_3);
        store_frag!(row_top, col2, at_n2_0, at_n2_1, at_n2_2, at_n2_3);
        store_frag!(row_top, col3, at_n3_0, at_n3_1, at_n3_2, at_n3_3);
        store_frag!(row_bot, col0, ab_n0_0, ab_n0_1, ab_n0_2, ab_n0_3);
        store_frag!(row_bot, col1, ab_n1_0, ab_n1_1, ab_n1_2, ab_n1_3);
        store_frag!(row_bot, col2, ab_n2_0, ab_n2_1, ab_n2_2, ab_n2_3);
        store_frag!(row_bot, col3, ab_n3_0, ab_n3_1, ab_n3_2, ab_n3_3);
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn lcg_seed(mut s: u64) -> impl FnMut() -> f32 {
    move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = ((s >> 33) as u32) as f32 / (u32::MAX as f32);
        v * 2.0 - 1.0
    }
}

fn make_inputs(m: usize, n: usize, k: usize, seed: u64) -> (Vec<u16>, Vec<u16>, Vec<bf16>, Vec<bf16>) {
    let mut rng_a = lcg_seed(seed);
    let mut rng_b = lcg_seed(seed.wrapping_add(0xdead_beef));
    let mut a_bf16 = Vec::with_capacity(m * k);
    let mut b_bf16 = Vec::with_capacity(k * n);
    for _ in 0..(m * k) {
        a_bf16.push(bf16::from_f32(rng_a()));
    }
    for _ in 0..(k * n) {
        b_bf16.push(bf16::from_f32(rng_b()));
    }
    let a_u16: Vec<u16> = a_bf16.iter().map(|v| v.to_bits()).collect();
    let b_u16: Vec<u16> = b_bf16.iter().map(|v| v.to_bits()).collect();
    (a_u16, b_u16, a_bf16, b_bf16)
}

fn cpu_gemm_ref(a: &[bf16], b: &[bf16], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += a[i * k + kk].to_f32() * b[kk * n + j].to_f32();
            }
            c[i * n + j] = acc;
        }
    }
    c
}

fn launch(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    m: usize,
    n: usize,
    k: usize,
    a_dev: &DeviceBuffer<u16>,
    b_dev: &DeviceBuffer<u16>,
    mut c_dev: &mut DeviceBuffer<f32>,
) -> Result<(), Box<dyn std::error::Error>> {
    let grid_x = ((n + 63) / 64) as u32;
    let grid_y = ((m + 63) / 64) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    let m_arg = m as u32;
    let n_arg = n as u32;
    let k_arg = k as u32;
    cuda_launch! {
        kernel: mma_gemm_kernel,
        stream: stream,
        module: module,
        config: cfg,
        args: [m_arg, n_arg, k_arg, slice(a_dev), slice(b_dev), slice_mut(c_dev)]
    }?;
    Ok(())
}

fn run_verify(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("\n--- VERIFY  M={} N={} K={} ---", m, n, k);
    assert!(k % 16 == 0, "K must be a multiple of 16");
    let (a_u16, b_u16, a_bf16, b_bf16) = make_inputs(m, n, k, seed);
    let a_dev = DeviceBuffer::from_host(stream, &a_u16)?;
    let b_dev = DeviceBuffer::from_host(stream, &b_u16)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(stream, m * n)?;

    launch(stream, module, m, n, k, &a_dev, &b_dev, &mut c_dev)?;
    stream.synchronize()?;
    let c_gpu = c_dev.to_host_vec(stream)?;
    let c_ref = cpu_gemm_ref(&a_bf16, &b_bf16, m, n, k);

    let atol = 2e-2f32;
    let rtol = 2e-2f32;
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut mismatches = 0usize;
    for i in 0..(m * n) {
        let g = c_gpu[i];
        let r = c_ref[i];
        let abs_err = (g - r).abs();
        let rel_err = if r.abs() > 1e-6 { abs_err / r.abs() } else { 0.0 };
        if abs_err > max_abs {
            max_abs = abs_err;
        }
        if rel_err > max_rel {
            max_rel = rel_err;
        }
        if abs_err > atol + rtol * r.abs() {
            mismatches += 1;
        }
    }
    println!(
        "  max_abs_err={:.4e}  max_rel_err={:.4e}  mismatches={}/{}",
        max_abs,
        max_rel,
        mismatches,
        m * n
    );
    if mismatches == 0 {
        println!("  PASS");
        Ok(true)
    } else {
        println!("  FAIL");
        let pn = n.min(8);
        for i in 0..m.min(4) {
            for j in 0..pn {
                print!("    gpu[{},{}]={:8.3} ref={:8.3}", i, j, c_gpu[i * n + j], c_ref[i * n + j]);
            }
            println!();
        }
        Ok(false)
    }
}

fn run_bench(
    stream: &Arc<CudaStream>,
    module: &Arc<CudaModule>,
    m: usize,
    n: usize,
    k: usize,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- BENCH  M={} N={} K={} ---", m, n, k);
    let (a_u16, b_u16, _, _) = make_inputs(m, n, k, seed);
    let a_dev = DeviceBuffer::from_host(stream, &a_u16)?;
    let b_dev = DeviceBuffer::from_host(stream, &b_u16)?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(stream, m * n)?;

    for _ in 0..10 {
        launch(stream, module, m, n, k, &a_dev, &b_dev, &mut c_dev)?;
    }
    stream.synchronize()?;

    const ITERS: usize = 50;
    let mut times_ms = Vec::with_capacity(ITERS);
    for _ in 0..ITERS {
        let t0 = Instant::now();
        launch(stream, module, m, n, k, &a_dev, &b_dev, &mut c_dev)?;
        stream.synchronize()?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        times_ms.push(dt);
    }
    times_ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times_ms[ITERS / 2];
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let tflops = flops / (median / 1000.0) / 1e12;
    println!("  median: {:.3} ms   {:.2} TFLOPS bf16", median, tflops);
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== mma.sync m16n8k16 bf16->f32 GEMM (4-warp 64x64) ===");
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    let ptx_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("mma_gemm.ptx");
    let ptx_file = ptx_path.to_str().ok_or("PTX path is not valid UTF-8")?;
    let module = ctx.load_module_from_file(ptx_file)?;
    println!("PTX loaded from: {}", ptx_path.display());

    let mode = std::env::args().nth(1).unwrap_or_else(|| "all".to_string());
    let do_verify = matches!(mode.as_str(), "verify" | "all");
    let do_bench = matches!(mode.as_str(), "bench" | "all");

    let mut all_pass = true;
    if do_verify {
        for &(m, n, k) in &[(16usize, 16usize, 16usize), (128, 128, 128), (256, 256, 256)] {
            let pass = run_verify(&stream, &module, m, n, k, 42)?;
            all_pass &= pass;
        }
    }
    if do_bench {
        run_bench(&stream, &module, 1024, 1024, 1024, 7)?;
        run_bench(&stream, &module, 4096, 4096, 4096, 7)?;
    }
    if do_verify && !all_pass {
        std::process::exit(1);
    }
    Ok(())
}
