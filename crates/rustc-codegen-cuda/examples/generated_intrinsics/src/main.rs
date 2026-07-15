/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end smoke test for the generated low-level intrinsic surface.
//!
//! The kernel calls generated coordinate, lane-mask, and vote intrinsics
//! directly. This covers the raw path instead of only `cuda-device` wrappers.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};
use cuda_intrinsics::sreg::{
    block_dim_x, block_dim_y, block_dim_z, block_idx_x, block_idx_y, block_idx_z, grid_dim_x,
    grid_dim_y, grid_dim_z, lane_id, lanemask_eq, lanemask_ge, lanemask_gt, lanemask_le,
    lanemask_lt, thread_idx_x, thread_idx_y, thread_idx_z,
};
use cuda_intrinsics::warp::{
    active_mask, all_sync, any_sync, ballot_sync, match_all_i64_sync, match_all_sync,
    match_any_i64_sync, match_any_sync, uni_sync,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn record_row_major_volume_idx(mut output: DisjointSlice<u32>) {
        let lane = lane_id();
        let lt = lanemask_lt();
        let le = lanemask_le();
        let eq = lanemask_eq();
        let ge = lanemask_ge();
        let gt = lanemask_gt();
        let member_mask = lt | eq | gt;
        let masks_ok = ((lt | eq) == le)
            & ((gt | eq) == ge)
            & ((lt ^ ge) == u32::MAX)
            & ((le ^ gt) == u32::MAX)
            & (eq.count_ones() == 1);
        let group = lane / 4;
        let expected_group_mask = 0xfu32 << (group * 4);
        // High-only values catch an accidental 32-bit match lowering.
        let wide_group = (group as u64) << 32;
        let wide_lane = (lane as u64) << 32;
        let active = active_mask();

        // SAFETY: every lane in each full warp executes these calls with the
        // same full member mask and instruction sequence.
        let (all_ok, any_ok, ballot, uniform, any32, any64, all32, all64) = unsafe {
            (
                all_sync(member_mask, masks_ok),
                any_sync(member_mask, lane == 0),
                ballot_sync(member_mask, masks_ok),
                uni_sync(member_mask, masks_ok),
                match_any_sync(member_mask, group),
                match_any_i64_sync(member_mask, wide_group),
                match_all_sync(member_mask, 42),
                match_all_i64_sync(member_mask, wide_lane),
            )
        };
        let votes_ok = all_ok & any_ok & uniform & (ballot == member_mask);
        let matches_ok = (active == member_mask)
            & (any32 == expected_group_mask)
            & (any64 == expected_group_mask)
            & (all32 == member_mask)
            & (all64 == 0);

        let block_width = block_dim_x();
        let block_height = block_dim_y();
        let block_depth = block_dim_z();
        let grid_width = grid_dim_x() * block_width;
        let grid_height = grid_dim_y() * block_height;
        let grid_depth = grid_dim_z() * block_depth;
        let column = block_idx_x() * block_width + thread_idx_x();
        let row = block_idx_y() * block_height + thread_idx_y();
        let plane = block_idx_z() * block_depth + thread_idx_z();
        let row_major_idx = ((plane * grid_height + row) * grid_width + column) as usize;

        if votes_ok
            && matches_ok
            && column < grid_width
            && row < grid_height
            && plane < grid_depth
            && row_major_idx < output.len()
        {
            // SAFETY: the row-major volume formula assigns one unique output
            // slot to each launched thread. The grid and allocation checks
            // above cover every index used below.
            unsafe {
                // Store one-based values so the zero-filled allocation also
                // reveals a missing write at row-major index zero.
                *output.get_unchecked_mut(row_major_idx) = row_major_idx as u32 + 1;
            }
        }
    }
}

fn main() {
    const BLOCKS_X: u32 = 3;
    const BLOCKS_Y: u32 = 2;
    const BLOCKS_Z: u32 = 2;
    const THREADS_X: u32 = 8;
    const THREADS_Y: u32 = 4;
    const THREADS_Z: u32 = 2;
    const WIDTH: u32 = BLOCKS_X * THREADS_X;
    const HEIGHT: u32 = BLOCKS_Y * THREADS_Y;
    const DEPTH: u32 = BLOCKS_Z * THREADS_Z;
    const ELEMENTS: u32 = WIDTH * HEIGHT * DEPTH;

    let context = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = context.default_stream();
    let mut output =
        DeviceBuffer::<u32>::zeroed(&stream, ELEMENTS as usize).expect("failed to allocate output");

    let module = kernels::load(&context).expect("failed to load generated PTX");
    // SAFETY: the launch dimensions contain exactly ELEMENTS threads, and the
    // kernel's checked row-major mapping assigns each one a distinct element
    // in the live ELEMENTS-entry output allocation.
    unsafe {
        module
            .record_row_major_volume_idx(
                &stream,
                LaunchConfig {
                    grid_dim: (BLOCKS_X, BLOCKS_Y, BLOCKS_Z),
                    block_dim: (THREADS_X, THREADS_Y, THREADS_Z),
                    shared_mem_bytes: 0,
                },
                &mut output,
            )
            .expect("failed to launch thread-index kernel");
    }

    let actual = output
        .to_host_vec(&stream)
        .expect("failed to copy output to the host");
    let expected: Vec<u32> = (1..=ELEMENTS).collect();
    assert_eq!(actual, expected);

    println!("PASS: generated X/Y/Z-coordinate intrinsics produced every row-major volume index");
}
