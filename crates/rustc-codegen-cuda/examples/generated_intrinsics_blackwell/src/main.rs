/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Compile coverage for generated high-target intrinsics.

use cuda_device::{
    DisjointSlice,
    barrier::Barrier,
    convert::{
        cvt_rn_relu_satfinite_tf32_f32, cvt_rn_relu_tf32_f32, cvt_rn_satfinite_tf32_f32,
        cvt_rn_tf32_f32, cvt_rna_satfinite_tf32_f32, cvt_rna_tf32_f32,
        cvt_rz_relu_satfinite_tf32_f32, cvt_rz_relu_tf32_f32, cvt_rz_satfinite_tf32_f32,
        cvt_rz_tf32_f32,
    },
    cuda_module, kernel, thread,
    tma::{self, TmaDescriptor},
};
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

    /// Keeps every generated TF32 conversion in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_tf32_conversions(mut output: DisjointSlice<u32>, value: f32) {
        let values = [
            cvt_rna_tf32_f32(value),
            cvt_rna_satfinite_tf32_f32(value),
            cvt_rn_tf32_f32(value),
            cvt_rn_relu_tf32_f32(value),
            cvt_rn_satfinite_tf32_f32(value),
            cvt_rn_relu_satfinite_tf32_f32(value),
            cvt_rz_tf32_f32(value),
            cvt_rz_relu_tf32_f32(value),
            cvt_rz_satfinite_tf32_f32(value),
            cvt_rz_relu_satfinite_tf32_f32(value),
        ];
        let start = thread::index_1d().get() * values.len();
        if start + values.len() <= output.len() {
            for (offset, converted) in values.into_iter().enumerate() {
                // SAFETY: the bounds check covers this thread's unique slots.
                unsafe { *output.get_unchecked_mut(start + offset) = converted };
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

    /// Compile-only coverage for the TMA compatibility API.
    #[kernel]
    pub unsafe fn compile_tma_compatibility(
        shared: *mut u8,
        tensor_map: *const TmaDescriptor,
        barrier: *mut Barrier,
        cta_mask: u16,
    ) {
        // This kernel is never launched with these placeholder addresses.
        unsafe {
            tma::cp_async_bulk_tensor_1d_g2s(shared, tensor_map, 0, barrier);
            tma::cp_async_bulk_tensor_2d_g2s(shared, tensor_map, 0, 0, barrier);
            tma::cp_async_bulk_tensor_2d_g2s_multicast(
                shared, tensor_map, 0, 0, barrier, cta_mask,
            );
            tma::cp_async_bulk_tensor_3d_g2s(shared, tensor_map, 0, 0, 0, barrier);
            tma::cp_async_bulk_tensor_4d_g2s(shared, tensor_map, 0, 0, 0, 0, barrier);
            tma::cp_async_bulk_tensor_5d_g2s(shared, tensor_map, 0, 0, 0, 0, 0, barrier);

            tma::cp_async_bulk_tensor_1d_s2g(shared, tensor_map, 0);
            tma::cp_async_bulk_tensor_2d_s2g(shared, tensor_map, 0, 0);
            tma::cp_async_bulk_tensor_3d_s2g(shared, tensor_map, 0, 0, 0);
            tma::cp_async_bulk_tensor_4d_s2g(shared, tensor_map, 0, 0, 0, 0);
            tma::cp_async_bulk_tensor_5d_s2g(shared, tensor_map, 0, 0, 0, 0, 0);
        }
        tma::cp_async_bulk_commit_group();
        tma::cp_async_bulk_wait_group(0);
        tma::cp_async_bulk_wait_group_read(0);
    }
}

fn main() {
    println!("PASS: generated Blackwell sparse MMA compile coverage");
}
