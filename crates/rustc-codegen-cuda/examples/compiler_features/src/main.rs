/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compiler Features Test - Testing multi-way match, enums, for loops, and more
//!
//! Tests:
//! - Multi-way match statements (integer switches)
//! - Enum support (Option<T>)
//! - For loops (range, break, continue, nested, iterators)
//! - Baseline tests (while loop, binary match, vecadd)
//! - Shared memory address casting
//! - 64-bit arithmetic
//! - Parallel for loop patterns
//!
//! Run: cargo oxide run compiler_features

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// PHASE 1: Multi-way Match (Integer Switches)
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// Test multi-way match on u32
    #[kernel]
    pub fn test_multiway_match_u32(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let result = match val {
                0 => 10u32,
                1 => 20u32,
                2 => 30u32,
                _ => 99u32,
            };
            *out_elem = result;
        }
    }

    // =============================================================================
    // PHASE 2: Enum Support - Option<T>
    // =============================================================================

    /// Test Option<T> - fundamental for for-loops
    #[kernel]
    pub fn test_option(val: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let maybe: Option<u32> = if val > 0 { Some(val) } else { None };
            let result = maybe.unwrap_or_default();
            *out_elem = result;
        }
    }

    // =============================================================================
    // PHASE 3: For Loops
    // =============================================================================

    /// Test simple for loop with range: sum of 0..8 = 28
    #[kernel]
    pub fn test_for_loop_sum(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..8 {
                sum += i;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with slice.iter()
    #[kernel]
    pub fn test_iter_sum(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for val in data.iter() {
                sum += *val;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with enumerate()
    #[kernel]
    pub fn test_enumerate(data: &[u32], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for (i, val) in data.iter().enumerate() {
                sum += (i as u32) * (*val);
            }
            *out_elem = sum;
        }
    }

    /// Test nested for loops: sum of i*j for i,j in 0..4 = 36
    #[kernel]
    pub fn test_nested_for_loops(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..4 {
                for j in 0u32..4 {
                    sum += i * j;
                }
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with early break: sum 0+1+2+3+4 = 10
    #[kernel]
    pub fn test_for_loop_break(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..100 {
                if i >= 5 {
                    break;
                }
                sum += i;
            }
            *out_elem = sum;
        }
    }

    /// Test for loop with continue: sum of odd numbers 1+3+5+7 = 16
    #[kernel]
    pub fn test_for_loop_continue(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            for i in 0u32..8 {
                if i % 2 == 0 {
                    continue;
                }
                sum += i;
            }
            *out_elem = sum;
        }
    }

    // =============================================================================
    // BASELINE TESTS
    // =============================================================================

    /// Baseline while loop for comparison (sum 0..8 = 28)
    #[kernel]
    pub fn baseline_while_loop(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let mut sum: u32 = 0;
            let mut i: u32 = 0;
            while i < 8 {
                sum += i;
                i += 1;
            }
            *out_elem = sum;
        }
    }

    /// Baseline: binary if-else
    #[kernel]
    pub fn baseline_binary_match(flag: bool, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let result = if flag { 100u32 } else { 0u32 };
            *out_elem = result;
        }
    }

    /// Baseline: simple arithmetic vecadd
    #[kernel]
    pub fn baseline_vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx_raw] + b[idx_raw];
        }
    }

    // =============================================================================
    // SHARED MEMORY ADDRESS CASTING TESTS
    // =============================================================================

    /// Test DIRECT cast to u64 - no intermediate pointer cast
    #[kernel]
    pub unsafe fn test_smem_addr_direct_u64(mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr = &raw const SMEM as u64;
            *out_elem = addr;
        }
    }

    /// Test Via *const u8 (current approach - has cvta round-trip)
    #[kernel]
    pub unsafe fn test_smem_addr_via_ptr_u8(mut out: DisjointSlice<u64>) {
        static mut SMEM: SharedArray<u8, 256, 128> = SharedArray::UNINIT;

        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr = &raw const SMEM as *const u8 as u64;
            *out_elem = addr;
        }
    }

    // =============================================================================
    // 64-BIT ARITHMETIC TESTS
    // =============================================================================

    /// Test 64-bit descriptor building - reproduces tcgen05 SMEM descriptor bug
    #[kernel]
    pub fn test_u64_descriptor_build(
        addr: u64,
        leading_dim_bytes: u32,
        stride_bytes: u32,
        mut out: DisjointSlice<u64>,
    ) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let addr_enc = (addr >> 4) & 0x3FFF;
            let ld_enc = ((leading_dim_bytes >> 4) & 0x3FFF) as u64;
            let stride_enc = ((stride_bytes >> 4) & 0x3FFF) as u64;
            let fixed_bit: u64 = 1u64 << 46;

            let desc = addr_enc | (ld_enc << 16) | (stride_enc << 32) | fixed_bit;
            *out_elem = desc;
        }
    }

    /// Simpler test: Just test that (val << 32) works correctly for 64-bit
    #[kernel]
    pub fn test_u64_shift_by_32(val: u64, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let shifted = val << 32;
            *out_elem = shifted;
        }
    }

    /// Test: (1u64 << 46) - fixed bit at position 46
    #[kernel]
    pub fn test_u64_shift_by_46(mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let fixed_bit: u64 = 1u64 << 46;
            *out_elem = fixed_bit;
        }
    }

    // =============================================================================
    // PHASE 4: Parallel For Loop Patterns
    // =============================================================================

    /// Parallel polynomial evaluation: p(x) = 1 + x + x^2 + ... + x^7
    #[kernel]
    pub fn parallel_polynomial_eval(input: &[f32], mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let x = input[idx_raw];
            let mut result: f32 = 0.0;
            let mut power: f32 = 1.0;
            for _ in 0u32..8 {
                result += power;
                power *= x;
            }
            *out_elem = result;
        }
    }

    /// Parallel chunked sum: each thread sums a contiguous chunk
    #[kernel]
    pub fn parallel_chunked_sum(data: &[u32], chunk_size: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let data_len = data.len() as u32;
            let mut sum: u32 = 0;

            for i in start..end {
                if i < data_len {
                    sum += data[i as usize];
                }
            }
            *out_elem = sum;
        }
    }

    /// Parallel local average: each thread computes average of a window
    #[kernel]
    pub fn parallel_local_average(data: &[f32], radius: u32, mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let pos = idx_raw as i32;
            let len = data.len() as i32;
            let r = radius as i32;

            let mut sum: f32 = 0.0;
            let mut count: u32 = 0;

            for offset in 0u32..(2 * radius + 1) {
                let sample_pos = pos - r + (offset as i32);
                if sample_pos >= 0 && sample_pos < len {
                    sum += data[sample_pos as usize];
                    count += 1;
                }
            }

            let avg = if count > 0 { sum / (count as f32) } else { 0.0 };
            *out_elem = avg;
        }
    }

    /// Parallel dot product contribution: each thread computes partial dot product
    #[kernel]
    pub fn parallel_dot_product_chunked(
        a: &[f32],
        b: &[f32],
        chunk_size: u32,
        mut out: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let len = a.len() as u32;

            let mut partial_sum: f32 = 0.0;
            for i in start..end {
                if i < len {
                    partial_sum += a[i as usize] * b[i as usize];
                }
            }
            *out_elem = partial_sum;
        }
    }

    /// Parallel matrix row sum: each thread sums one row
    #[kernel]
    pub fn parallel_row_sum(matrix: &[u32], cols: u32, mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let row = idx_raw as u32;
            let row_start = row * cols;

            let mut sum: u32 = 0;
            for col in 0u32..cols {
                let elem_idx = row_start + col;
                if (elem_idx as usize) < matrix.len() {
                    sum += matrix[elem_idx as usize];
                }
            }
            *out_elem = sum;
        }
    }

    /// Parallel histogram counting: count occurrences in range [low, high)
    #[kernel]
    pub fn parallel_range_count(
        data: &[u32],
        chunk_size: u32,
        low: u32,
        high: u32,
        mut out: DisjointSlice<u32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let start = idx_raw as u32 * chunk_size;
            let end = start + chunk_size;
            let len = data.len() as u32;

            let mut count: u32 = 0;
            for i in start..end {
                if i < len {
                    let val = data[i as usize];
                    if val >= low && val < high {
                        count += 1;
                    }
                }
            }
            *out_elem = count;
        }
    }

    /// Parallel partial product: each thread computes a factorial-like product
    #[kernel]
    pub fn parallel_partial_product(elements_per_thread: u32, mut out: DisjointSlice<u64>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            let base = idx_raw as u64 * elements_per_thread as u64;

            let mut product: u64 = 1;
            for i in 1u32..=elements_per_thread {
                product *= base + (i as u64);
            }
            *out_elem = product;
        }
    }
}

// =============================================================================
// HOST CODE
// =============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Compiler Features Test (Unified) ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let module = ctx.load_module_from_file("compiler_features.ptx")?;
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    const N: usize = 1;
    let cfg = LaunchConfig::for_num_elems(N as u32);

    // Test baseline while loop
    println!("Testing: baseline_while_loop");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.baseline_while_loop((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 28, "baseline_while_loop failed");
        println!("  ✓ Result: {} (expected 28)", result[0]);
    }

    // Test binary match
    println!("Testing: baseline_binary_match");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.baseline_binary_match((stream).as_ref(), cfg, true, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 100, "baseline_binary_match(true) failed");
        println!("  ✓ flag=true: {} (expected 100)", result[0]);

        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.baseline_binary_match((stream).as_ref(), cfg, false, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 0, "baseline_binary_match(false) failed");
        println!("  ✓ flag=false: {} (expected 0)", result[0]);
    }

    // Test vecadd
    println!("Testing: baseline_vecadd");
    {
        let a = vec![1.0f32, 2.0, 3.0, 4.0];
        let b = vec![10.0f32, 20.0, 30.0, 40.0];
        let n = a.len();

        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;

        module.baseline_vecadd(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(n as u32),
            &a_dev,
            &b_dev,
            &mut c_dev,
        )?;
        let result = c_dev.to_host_vec(&stream)?;
        let expected = vec![11.0f32, 22.0, 33.0, 44.0];
        assert_eq!(result, expected, "baseline_vecadd failed");
        println!("  ✓ Result: {:?}", result);
    }

    // Test multi-way match
    println!("Testing: test_multiway_match_u32");
    {
        let test_cases = [(0u32, 10u32), (1, 20), (2, 30), (3, 99), (100, 99)];
        for (val, expected) in test_cases {
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
            module.test_multiway_match_u32((stream).as_ref(), cfg, val, &mut out_dev)?;
            let result = out_dev.to_host_vec(&stream)?;
            assert_eq!(
                result[0], expected,
                "test_multiway_match_u32({}) failed",
                val
            );
            println!("  ✓ val={}: {} (expected {})", val, result[0], expected);
        }
    }

    // Test Option<T> enum
    println!("Testing: test_option");
    {
        let test_cases = [(0u32, 0u32), (1u32, 1u32), (42u32, 42u32)];
        for (val, expected) in test_cases {
            let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
            module.test_option((stream).as_ref(), cfg, val, &mut out_dev)?;
            let result = out_dev.to_host_vec(&stream)?;
            assert_eq!(result[0], expected, "test_option({}) failed", val);
            println!("  ✓ val={}: {} (expected {})", val, result[0], expected);
        }
    }

    // Test for loop sum
    println!("Testing: test_for_loop_sum");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_for_loop_sum((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 28, "test_for_loop_sum failed");
        println!("  ✓ Result: {} (expected 28)", result[0]);
    }

    // Test iter sum
    println!("Testing: test_iter_sum");
    {
        let data = vec![1u32, 2, 3, 4, 5];
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_iter_sum((stream).as_ref(), cfg, &data_dev, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 15, "test_iter_sum failed");
        println!("  ✓ Result: {} (expected 15)", result[0]);
    }

    // Test enumerate
    println!("Testing: test_enumerate");
    {
        let data = vec![10u32, 20, 30, 40]; // 0*10 + 1*20 + 2*30 + 3*40 = 0+20+60+120=200
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_enumerate((stream).as_ref(), cfg, &data_dev, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 200, "test_enumerate failed");
        println!("  ✓ Result: {} (expected 200)", result[0]);
    }

    // Test for loop break
    println!("Testing: test_for_loop_break");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_for_loop_break((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 10, "test_for_loop_break failed");
        println!("  ✓ Result: {} (expected 10)", result[0]);
    }

    // Test for loop continue
    println!("Testing: test_for_loop_continue");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_for_loop_continue((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 16, "test_for_loop_continue failed");
        println!("  ✓ Result: {} (expected 16)", result[0]);
    }

    // Test nested for loops
    println!("Testing: test_nested_for_loops");
    {
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N)?;
        module.test_nested_for_loops((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 36, "test_nested_for_loops failed");
        println!("  ✓ Result: {} (expected 36)", result[0]);
    }

    // Test u64 shift by 32
    println!("Testing: test_u64_shift_by_32");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        module.test_u64_shift_by_32((stream).as_ref(), cfg, 8u64, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 8u64 << 32;
        assert_eq!(result[0], expected, "test_u64_shift_by_32 failed");
        println!(
            "  ✓ Result: 0x{:016X} (expected 0x{:016X})",
            result[0], expected
        );
    }

    // Test u64 shift by 46
    println!("Testing: test_u64_shift_by_46");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, N)?;
        module.test_u64_shift_by_46((stream).as_ref(), cfg, &mut out_dev)?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 1u64 << 46;
        assert_eq!(result[0], expected, "test_u64_shift_by_46 failed");
        println!(
            "  ✓ Result: 0x{:016X} (expected 0x{:016X})",
            result[0], expected
        );
    }

    // Test parallel polynomial eval
    println!("Testing: parallel_polynomial_eval");
    {
        let input = vec![2.0f32; 4]; // p(2) = 1 + 2 + 4 + 8 + 16 + 32 + 64 + 128 = 255
        let input_dev = DeviceBuffer::from_host(&stream, &input)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, 4)?;
        module.parallel_polynomial_eval(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(4),
            &input_dev,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        let expected = 255.0f32;
        assert!(
            (result[0] - expected).abs() < 0.01,
            "parallel_polynomial_eval failed"
        );
        println!("  ✓ Result: {} (expected {})", result[0], expected);
    }

    // Test parallel chunked sum
    println!("Testing: parallel_chunked_sum");
    {
        let data: Vec<u32> = (1..=16).collect(); // 1,2,3,...,16
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
        // 4 threads, chunk_size=4: thread 0 sums 1+2+3+4=10, thread 1 sums 5+6+7+8=26, etc.
        module.parallel_chunked_sum(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(4),
            &data_dev,
            4u32,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 10, "parallel_chunked_sum[0] failed");
        assert_eq!(result[1], 26, "parallel_chunked_sum[1] failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel row sum
    println!("Testing: parallel_row_sum");
    {
        // 4x4 matrix with row i having values i*4+1, i*4+2, i*4+3, i*4+4
        let matrix: Vec<u32> = (1..=16).collect();
        let matrix_dev = DeviceBuffer::from_host(&stream, &matrix)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4)?;
        module.parallel_row_sum(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(4),
            &matrix_dev,
            4u32,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        // Row 0: 1+2+3+4=10, Row 1: 5+6+7+8=26, Row 2: 9+10+11+12=42, Row 3: 13+14+15+16=58
        assert_eq!(result, vec![10, 26, 42, 58], "parallel_row_sum failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel partial product
    println!("Testing: parallel_partial_product");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 4)?;
        // Thread 0: product of 1,2,3 = 6
        // Thread 1: product of 4,5,6 = 120
        // Thread 2: product of 7,8,9 = 504
        // Thread 3: product of 10,11,12 = 1320
        module.parallel_partial_product(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(4),
            3u32,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        assert_eq!(result[0], 6, "parallel_partial_product[0] failed");
        assert_eq!(result[1], 120, "parallel_partial_product[1] failed");
        println!("  ✓ Results: {:?}", result);
    }

    // Test parallel local average
    println!("Testing: parallel_local_average");
    {
        const N: usize = 512;
        const RADIUS: u32 = 3;
        let data: Vec<f32> = (0..N).map(|i| i as f32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
        module.parallel_local_average(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(N as u32),
            &data_dev,
            RADIUS,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        // At position 256 (middle), average of 253..259 = 256.0
        let mid = N / 2;
        let expected = mid as f32;
        let tol = 0.001;
        assert!(
            (result[mid] - expected).abs() < tol,
            "parallel_local_average[{}] failed: got {}, expected {}",
            mid,
            result[mid],
            expected
        );
        println!(
            "  ✓ result[{}]={:.2} (expected {:.2})",
            mid, result[mid], expected
        );
    }

    // Test parallel dot product chunked
    println!("Testing: parallel_dot_product_chunked");
    {
        const NUM_THREADS: usize = 128;
        const CHUNK_SIZE: u32 = 32;
        const TOTAL_SIZE: usize = NUM_THREADS * CHUNK_SIZE as usize;
        // a = [1, 1, 1, ...], b = [2, 2, 2, ...], each element contributes 2.0
        let a: Vec<f32> = vec![1.0f32; TOTAL_SIZE];
        let b: Vec<f32> = vec![2.0f32; TOTAL_SIZE];
        let a_dev = DeviceBuffer::from_host(&stream, &a)?;
        let b_dev = DeviceBuffer::from_host(&stream, &b)?;
        let mut out_dev = DeviceBuffer::<f32>::zeroed(&stream, NUM_THREADS)?;
        module.parallel_dot_product_chunked(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(NUM_THREADS as u32),
            &a_dev,
            &b_dev,
            CHUNK_SIZE,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        // Each thread: chunk_size * 2.0 = 64.0, total = 128 * 64 = 8192.0
        let total: f32 = result.iter().sum();
        let expected_total = (TOTAL_SIZE as f32) * 2.0;
        let tol = 0.1;
        assert!(
            (total - expected_total).abs() < tol,
            "parallel_dot_product_chunked total failed: got {}, expected {}",
            total,
            expected_total
        );
        println!(
            "  ✓ Total dot product: {} (expected {}), each thread: {}",
            total, expected_total, result[0]
        );
    }

    // Test parallel range count
    println!("Testing: parallel_range_count");
    {
        const NUM_THREADS: usize = 256;
        const CHUNK_SIZE: u32 = 16;
        const TOTAL_SIZE: usize = NUM_THREADS * CHUNK_SIZE as usize;
        let data: Vec<u32> = (0..TOTAL_SIZE as u32).collect();
        let data_dev = DeviceBuffer::from_host(&stream, &data)?;
        let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, NUM_THREADS)?;
        let low: u32 = 50;
        let high: u32 = 150;
        module.parallel_range_count(
            (stream).as_ref(),
            LaunchConfig::for_num_elems(NUM_THREADS as u32),
            &data_dev,
            CHUNK_SIZE,
            low,
            high,
            &mut out_dev,
        )?;
        let result = out_dev.to_host_vec(&stream)?;
        // Count values in [50, 150) = 100 values
        let total: u32 = result.iter().sum();
        let expected = 100u32;
        assert_eq!(
            total, expected,
            "parallel_range_count total failed: got {}, expected {}",
            total, expected
        );
        println!(
            "  ✓ Total count in [50, 150): {} (expected {})",
            total, expected
        );
    }

    // ==========================================================================
    // SHARED MEMORY ADDRESS CASTING TESTS
    // ==========================================================================
    println!("\n-----------------------------------------");
    println!("SHARED MEMORY ADDRESS CASTING TESTS");
    println!("-----------------------------------------");
    println!("Comparing: &raw const SMEM as u64  vs  &raw const SMEM as *const u8 as u64");
    println!();

    let mut smem_direct_ok = true;

    // Test 1: DIRECT cast - &raw const SMEM as u64.
    // This is a real pass/fail test: the direct cast must keep the address
    // in the shared address space (small value, no cvta round-trip).
    println!("Testing: test_smem_addr_direct_u64 (&raw const SMEM as u64)");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.test_smem_addr_direct_u64(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(1),
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        println!("  DIRECT addr: 0x{:016x}", result[0]);
        if result[0] < 0x100000 {
            println!("  ✓ SMALL value = likely shared address (what we want!)");
        } else {
            println!("  ✗ FAILED: LARGE value = generic address (cvta happened)");
            smem_direct_ok = false;
        }
    }

    // Test 2: Via *const u8 (informational, documents known limitation).
    // The intermediate `*const u8` cast currently triggers a cvta round-trip;
    // we print the observed address but don't fail on it. Whenever this lights
    // up as "SMALL value" we know the upstream codegen quirk is gone and we
    // can promote it to a real assertion.
    println!("\nInfo: test_smem_addr_via_ptr_u8 (&raw const SMEM as *const u8 as u64)");
    println!("  (known: today's lowering inserts a cvta round-trip here)");
    {
        let mut out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1)?;
        unsafe {
            module.test_smem_addr_via_ptr_u8(
                (stream).as_ref(),
                LaunchConfig::for_num_elems(1),
                &mut out_dev,
            )
        }?;
        let result = out_dev.to_host_vec(&stream)?;
        println!("  VIA PTR addr: 0x{:016x}", result[0]);
        if result[0] < 0x100000 {
            println!("  (small value — cvta round-trip elided)");
        } else {
            println!("  (large value — cvta round-trip present, as documented)");
        }
    }

    println!("\n-----------------------------------------");
    println!("Check PTX for 'cvta' in each test function.");

    if !smem_direct_ok {
        println!("\n=== FAILED: at least one test did not pass ===");
        std::process::exit(1);
    }

    println!("\n=== ALL TESTS PASSED ✓ ===");
    Ok(())
}
