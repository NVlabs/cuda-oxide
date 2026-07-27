/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use cuda_device::thread::{LaunchContextRef, __internal};
use cuda_device::{DisjointSlice, RuntimeRowMajorTiles, Uniform};

fn cannot_mint_two_tiles_from_one_coordinate<'kernel>(
    launch_context: LaunchContextRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
    mut c: DisjointSlice<'_, f32, RuntimeRowMajorTiles<1, 1>>,
) {
    let coord = __internal::coord_2d_u32(launch_context);
    // SAFETY: a literal is trivially the same in every thread.
    let stride = unsafe { Uniform::new_unchecked(64) };
    let _first = c.tile_2d32_rt(coord, stride);
    let _second = c.tile_2d32_rt(coord, stride);
}

fn main() {}
