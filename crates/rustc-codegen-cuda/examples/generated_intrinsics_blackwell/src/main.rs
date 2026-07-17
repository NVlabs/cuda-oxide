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
    cvt_rn_satfinite_e4m3x2_f32, cvt_rn_satfinite_e5m2x2_f32, cvt_rn_satfinite_relu_e4m3x2_f32,
    cvt_rn_satfinite_relu_e5m2x2_f32,
};
use cuda_intrinsics::matrix;

#[cuda_module]
mod kernels {
    use super::*;

    /// Keeps every generated packed-FP8 conversion in device code.
    #[kernel]
    pub fn compile_fp8_conversions(mut output: DisjointSlice<u16>, low: f32, high: f32) {
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

    /// Keeps the complete ordered sparse F16 matrix in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_ordered_f8f6f4_f16(mut output: DisjointSlice<u32>) {
        let c = [0; 2];
        let a = [0; 4];
        let b = [0; 4];
        let metadata = 0x4444_4444;

        // SAFETY: every lane follows the same warp-synchronous sequence. The
        // selector and ordered metadata use their only admitted forms.
        let values = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m1_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e2m3_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e3m2_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e4m3_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e2m1_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e2m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e3m2_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e4m3_f16(
                    c, a, b, metadata, 0,
                ),
                matrix::mma_sp_ordered_metadata_m16n8k64_kind_f8f6f4_f16_e5m2_e5m2_f16(
                    c, a, b, metadata, 0,
                ),
            ]
        };
        let mut value = 0;
        for lanes in values {
            value ^= lanes[0] ^ lanes[1];
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every dense `kind::f8f6f4` F32 MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_dense_f8f6f4_f32(mut output: DisjointSlice<f32>) {
        let c = [0.0; 4];
        let a = [0; 4];
        let b = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let value = unsafe {
            matrix::mma_m16n8k32_f32_e2m1_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m1_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e2m3_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e3m2_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e4m3_e5m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e2m1(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e2m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e3m2(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e4m3(c, a, b)[0]
                + matrix::mma_m16n8k32_f32_e5m2_e5m2(c, a, b)[0]
        };

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every dense `kind::f8f6f4` F16 MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_dense_f8f6f4_f16(mut output: DisjointSlice<u32>) {
        let c = [0; 2];
        let a = [0; 4];
        let b = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let values = unsafe {
            [
                matrix::mma_m16n8k32_f16_e2m1_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m1_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e2m3_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e3m2_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e4m3_e5m2(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e2m1(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e2m3(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e3m2(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e4m3(c, a, b),
                matrix::mma_m16n8k32_f16_e5m2_e5m2(c, a, b),
            ]
        };
        let mut value = 0;
        for lanes in values {
            value ^= lanes[0] ^ lanes[1];
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every standard FP8 register-MMA form in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub fn compile_standard_fp8_mma(mut output: DisjointSlice<u32>) {
        let c_f16 = [0; 2];
        let c_f32 = [0.0; 4];
        let a_k16 = [0; 2];
        let b_k16 = 0;
        let a_k32 = [0; 4];
        let b_k32 = [0; 2];

        // SAFETY: every lane follows the same warp-synchronous sequence.
        let f16_values = unsafe {
            [
                matrix::mma_m16n8k16_fp8_f16_e4m3_e4m3(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e4m3_e5m2(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e5m2_e4m3(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f16_e5m2_e5m2(c_f16, a_k16, b_k16),
                matrix::mma_m16n8k32_fp8_f16_e4m3_e4m3(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e4m3_e5m2(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e5m2_e4m3(c_f16, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f16_e5m2_e5m2(c_f16, a_k32, b_k32),
            ]
        };
        let f32_values = unsafe {
            [
                matrix::mma_m16n8k16_fp8_f32_e4m3_e4m3(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e4m3_e5m2(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e5m2_e4m3(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k16_fp8_f32_e5m2_e5m2(c_f32, a_k16, b_k16),
                matrix::mma_m16n8k32_fp8_f32_e4m3_e4m3(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e4m3_e5m2(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e5m2_e4m3(c_f32, a_k32, b_k32),
                matrix::mma_m16n8k32_fp8_f32_e5m2_e5m2(c_f32, a_k32, b_k32),
            ]
        };

        let mut value = 0;
        for lanes in f16_values {
            value ^= lanes[0] ^ lanes[1];
        }
        for lanes in f32_values {
            value ^= lanes[0].to_bits() ^ lanes[1].to_bits();
            value ^= lanes[2].to_bits() ^ lanes[3].to_bits();
        }

        if let Some((slot, _)) = output.get_mut_indexed() {
            *slot = value;
        }
    }

    /// Keeps every Blackwell `ldmatrix` variant in device code.
    ///
    /// This kernel is compile-only and is never launched by the example.
    #[kernel]
    pub unsafe fn compile_blackwell_ldmatrix(input: *const u8, output: *mut u32) {
        // SAFETY: every lane follows the same sequence. A real caller must
        // provide 16-byte-aligned shared addresses with 32 readable bytes.
        let values = unsafe {
            [
                matrix::ldmatrix_m16n16_x1_trans_b8(input)[0],
                matrix::ldmatrix_m16n16_x1_trans_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m16n16_x1_trans_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m16n16_x2_trans_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m8n16_x1_b8x16_b4x16_p64(input),
                matrix::ldmatrix_m8n16_x1_b8x16_b6x16_p32(input),
                matrix::ldmatrix_m8n16_x2_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m8n16_x2_b8x16_b6x16_p32(input)[0],
                matrix::ldmatrix_m8n16_x4_b8x16_b4x16_p64(input)[0],
                matrix::ldmatrix_m8n16_x4_b8x16_b6x16_p32(input)[0],
            ]
        };

        for (index, value) in values.into_iter().enumerate() {
            // SAFETY: a real caller must provide space for all 12 results.
            unsafe { output.add(index).write(value) };
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
            tma::cp_async_bulk_tensor_2d_g2s_multicast(shared, tensor_map, 0, 0, barrier, cta_mask);
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
