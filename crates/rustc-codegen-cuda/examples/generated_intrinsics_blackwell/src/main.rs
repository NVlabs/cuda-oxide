/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile coverage for generated high-target intrinsics.

use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cuda_intrinsics::convert::{
    cvt_rn_satfinite_e4m3x2_f32, cvt_rn_satfinite_e5m2x2_f32,
    cvt_rn_satfinite_relu_e4m3x2_f32, cvt_rn_satfinite_relu_e5m2x2_f32,
};
use cuda_intrinsics::matrix;

#[cuda_module]
mod kernels {
    use super::*;

    /// Keeps every generated packed-FP8 conversion in device code.
    #[kernel]
    pub fn compile_fp8_conversions(
        mut output: DisjointSlice<u16>,
        low: f32,
        high: f32,
    ) {
        let values = [
            cvt_rn_satfinite_e4m3x2_f32(low, high),
            cvt_rn_satfinite_relu_e4m3x2_f32(low, high),
            cvt_rn_satfinite_e5m2x2_f32(low, high),
            cvt_rn_satfinite_relu_e5m2x2_f32(low, high),
        ];
        let start = thread::index_1d().get() * values.len();
        if start + values.len() <= output.len() {
            for (offset, value) in values.into_iter().enumerate() {
                // SAFETY: the bounds check covers this thread's unique slots.
                unsafe { *output.get_unchecked_mut(start + offset) = value };
            }
        }
    }

    /// Keeps the complete ordered `kind::f8f6f4` F32 matrix in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_ordered_f8f6f4_f32(mut output: DisjointSlice<f32>) {
        let c = [0.0; 4];
        let a = [0; 4];
        let b = [0; 4];
        let metadata = 0x4444_4444;

        // SAFETY: every lane follows the same warp-synchronous sequence. The
        // selector and ordered metadata use their only admitted forms.
        let value = unsafe {
            matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m1_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e2m3_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e3m2_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e4m3_e5m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e2m1_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e2m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e3m2_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e4m3_f32(
                c, a, b, metadata, 0,
            )[0] + matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f32_e5m2_e5m2_f32(
                c, a, b, metadata, 0,
            )[0]
        };

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }
}

fn main() {
    println!("PASS: generated Blackwell sparse MMA compile coverage");
}
