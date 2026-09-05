/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile-only reproduction: a TMA G2S call makes ptxas ignore setmaxnreg.

use cuda_device::{
    SharedArray,
    barrier::{Barrier, mbarrier_init},
    kernel, launch_bounds, thread,
    tma::{
        TmaDescriptor, cp_async_bulk_tensor_1d_g2s_cta, cp_async_bulk_tensor_2d_g2s_cta,
        cp_async_bulk_tensor_3d_g2s_cta, cp_async_bulk_tensor_4d_g2s_cta,
        cp_async_bulk_tensor_5d_g2s_cta,
    },
};
use cuda_host::cuda_module;

const THREADS: u32 = 384;
const MATH_THREADS: u32 = 256;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(THREADS, 1)]
    pub unsafe fn setmaxnreg_control(output: *mut u32) {
        let thread = thread::threadIdx_x();
        if thread >= MATH_THREADS {
            unsafe { thread::setmaxnreg_dec::<40>() };
            return;
        }
        unsafe { thread::setmaxnreg_inc::<232>() };
        if thread == 0 {
            unsafe { output.write(1) };
        }
    }

    #[kernel]
    #[launch_bounds(THREADS, 1)]
    pub unsafe fn setmaxnreg_with_tma(tensor_map: *const TmaDescriptor, output: *mut u32) {
        static mut TILE: SharedArray<u8, 128, 128> = SharedArray::UNINIT;
        static mut READY: Barrier = Barrier::UNINIT;

        let thread = thread::threadIdx_x();
        if thread == 0 {
            unsafe { mbarrier_init(&raw mut READY, 1) };
        }
        thread::sync_threads();
        if thread >= MATH_THREADS {
            unsafe { thread::setmaxnreg_dec::<40>() };
            if thread == MATH_THREADS {
                unsafe {
                    cp_async_bulk_tensor_2d_g2s_cta(
                        (&raw mut TILE).cast(),
                        tensor_map,
                        0,
                        0,
                        &raw mut READY,
                    );
                }
            }
            return;
        }
        unsafe { thread::setmaxnreg_inc::<232>() };
        if thread == 0 {
            unsafe { output.write(1) };
        }
    }

    #[kernel]
    #[launch_bounds(32, 1)]
    pub unsafe fn cta_g2s_all_dimensions(tensor_map: *const TmaDescriptor) {
        static mut TILE: SharedArray<u8, 128, 128> = SharedArray::UNINIT;
        static mut READY: Barrier = Barrier::UNINIT;

        if thread::threadIdx_x() == 0 {
            unsafe {
                mbarrier_init(&raw mut READY, 1);
                let tile = (&raw mut TILE).cast();
                let barrier = &raw mut READY;
                cp_async_bulk_tensor_1d_g2s_cta(tile, tensor_map, 0, barrier);
                cp_async_bulk_tensor_2d_g2s_cta(tile, tensor_map, 0, 0, barrier);
                cp_async_bulk_tensor_3d_g2s_cta(tile, tensor_map, 0, 0, 0, barrier);
                cp_async_bulk_tensor_4d_g2s_cta(tile, tensor_map, 0, 0, 0, 0, barrier);
                cp_async_bulk_tensor_5d_g2s_cta(tile, tensor_map, 0, 0, 0, 0, 0, barrier);
            }
        }
    }
}

fn main() {}
