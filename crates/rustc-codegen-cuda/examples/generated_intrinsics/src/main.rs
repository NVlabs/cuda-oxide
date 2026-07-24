/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end smoke test for the generated low-level intrinsic surface.
//!
//! The kernel calls generated coordinate, lane-mask, vote, and shuffle intrinsics
//! directly. This covers the raw path instead of only `cuda-device` wrappers.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel};
use cuda_device::{bf16x2 as device_bf16x2, f16x2 as device_f16x2, float as device_float};
use cuda_intrinsics::matrix;
use cuda_intrinsics::prmt::{prmt, prmt_b4e, prmt_ecl, prmt_ecr, prmt_f4e, prmt_rc8, prmt_rc16};
use cuda_intrinsics::sreg::{
    block_dim_x, block_dim_y, block_dim_z, block_idx_x, block_idx_y, block_idx_z, grid_dim_x,
    grid_dim_y, grid_dim_z, lane_id, lanemask_eq, lanemask_ge, lanemask_gt, lanemask_le,
    lanemask_lt, thread_idx_x, thread_idx_y, thread_idx_z,
};
use cuda_intrinsics::warp::{
    active_mask, all_sync, any_sync, ballot_sync, match_all_i64_sync, match_all_sync,
    match_any_i64_sync, match_any_sync, shuffle_down_f32_sync, shuffle_down_sync,
    shuffle_down_u64_sync, shuffle_f32_sync, shuffle_sync, shuffle_u64_sync, shuffle_up_f32_sync,
    shuffle_up_sync, shuffle_up_u64_sync, shuffle_xor_f32_sync, shuffle_xor_sync,
    shuffle_xor_u64_sync, sync_mask, uni_sync,
};
use cuda_intrinsics::{bf16x2 as raw_bf16x2, f16x2 as raw_f16x2, float as raw_float};

#[cuda_module]
mod kernels {
    use super::*;

    /// Keeps every explicit-rounding scalar arithmetic intrinsic in the module.
    ///
    /// This coverage kernel is compiled but not launched by the example.
    #[allow(clippy::too_many_arguments)]
    #[kernel]
    pub fn compile_scalar_explicit_rounding(
        mut output_f32: DisjointSlice<f32>,
        mut output_f64: DisjointSlice<f64>,
        a32: f32,
        b32: f32,
        c32: f32,
        a64: f64,
        b64: f64,
        c64: f64,
    ) {
        if cuda_device::thread::index_1d().get() != 0 {
            return;
        }
        let values_f64 = [
            raw_float::mul_rn_f64(a64, b64),
            device_float::mul_rz_f64(a64, b64),
            raw_float::mul_rm_f64(a64, b64),
            device_float::mul_rp_f64(a64, b64),
            raw_float::div_rn_f64(a64, b64),
            device_float::div_rz_f64(a64, b64),
            raw_float::div_rm_f64(a64, b64),
            device_float::div_rp_f64(a64, b64),
            raw_float::fma_rn_f64(a64, b64, c64),
            device_float::fma_rz_f64(a64, b64, c64),
            raw_float::fma_rm_f64(a64, b64, c64),
            device_float::fma_rp_f64(a64, b64, c64),
            raw_float::add_rn_f64(a64, b64),
            device_float::add_rz_f64(a64, b64),
            raw_float::add_rm_f64(a64, b64),
            device_float::add_rp_f64(a64, b64),
        ];
        let values_f32 = [
            raw_float::mul_rn_f32(a32, b32),
            raw_float::mul_rn_ftz_f32(a32, b32),
            device_float::mul_rz_f32(a32, b32),
            device_float::mul_rz_ftz_f32(a32, b32),
            raw_float::mul_rm_f32(a32, b32),
            raw_float::mul_rm_ftz_f32(a32, b32),
            device_float::mul_rp_f32(a32, b32),
            device_float::mul_rp_ftz_f32(a32, b32),
            raw_float::div_rn_f32(a32, b32),
            raw_float::div_rn_ftz_f32(a32, b32),
            device_float::div_rz_f32(a32, b32),
            device_float::div_rz_ftz_f32(a32, b32),
            raw_float::div_rm_f32(a32, b32),
            raw_float::div_rm_ftz_f32(a32, b32),
            device_float::div_rp_f32(a32, b32),
            device_float::div_rp_ftz_f32(a32, b32),
            raw_float::fma_rn_f32(a32, b32, c32),
            raw_float::fma_rn_ftz_f32(a32, b32, c32),
            raw_float::fma_rn_sat_f32(a32, b32, c32),
            raw_float::fma_rn_ftz_sat_f32(a32, b32, c32),
            device_float::fma_rz_f32(a32, b32, c32),
            device_float::fma_rz_ftz_f32(a32, b32, c32),
            device_float::fma_rz_sat_f32(a32, b32, c32),
            device_float::fma_rz_ftz_sat_f32(a32, b32, c32),
            raw_float::fma_rm_f32(a32, b32, c32),
            raw_float::fma_rm_ftz_f32(a32, b32, c32),
            raw_float::fma_rm_sat_f32(a32, b32, c32),
            raw_float::fma_rm_ftz_sat_f32(a32, b32, c32),
            device_float::fma_rp_f32(a32, b32, c32),
            device_float::fma_rp_ftz_f32(a32, b32, c32),
            device_float::fma_rp_sat_f32(a32, b32, c32),
            device_float::fma_rp_ftz_sat_f32(a32, b32, c32),
            raw_float::add_rn_f32(a32, b32),
            raw_float::add_rn_ftz_f32(a32, b32),
            raw_float::add_rn_sat_f32(a32, b32),
            raw_float::add_rn_ftz_sat_f32(a32, b32),
            device_float::add_rz_f32(a32, b32),
            device_float::add_rz_ftz_f32(a32, b32),
            device_float::add_rz_sat_f32(a32, b32),
            device_float::add_rz_ftz_sat_f32(a32, b32),
            raw_float::add_rm_f32(a32, b32),
            raw_float::add_rm_ftz_f32(a32, b32),
            raw_float::add_rm_sat_f32(a32, b32),
            raw_float::add_rm_ftz_sat_f32(a32, b32),
            device_float::add_rp_f32(a32, b32),
            device_float::add_rp_ftz_f32(a32, b32),
            device_float::add_rp_sat_f32(a32, b32),
            device_float::add_rp_ftz_sat_f32(a32, b32),
        ];
        for (index, value) in values_f32.into_iter().enumerate() {
            if index < output_f32.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_f32.get_unchecked_mut(index) = value };
            }
        }
        for (index, value) in values_f64.into_iter().enumerate() {
            if index < output_f64.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_f64.get_unchecked_mut(index) = value };
            }
        }
    }

    /// Keeps every generated scalar math intrinsic in the module.
    ///
    /// This coverage kernel is compiled but not launched by the example. It
    /// exercises both lowering paths of the scalar_math family: typed
    /// `llvm.nvvm.*` calls (rcp/sqrt) and inline PTX (sin/cos/lg2/rsqrt).
    #[kernel]
    pub fn compile_scalar_math(
        mut output_f32: DisjointSlice<f32>,
        mut output_f64: DisjointSlice<f64>,
        a32: f32,
        a64: f64,
    ) {
        if cuda_device::thread::index_1d().get() != 0 {
            return;
        }
        let values_f32 = [
            raw_float::sin_approx_f32(a32),
            device_float::sin_approx_ftz_f32(a32),
            raw_float::cos_approx_f32(a32),
            device_float::cos_approx_ftz_f32(a32),
            raw_float::lg2_approx_f32(a32),
            device_float::lg2_approx_ftz_f32(a32),
            raw_float::rcp_approx_ftz_f32(a32),
            device_float::rcp_rn_f32(a32),
            raw_float::rcp_rn_ftz_f32(a32),
            device_float::rcp_rz_f32(a32),
            raw_float::rcp_rz_ftz_f32(a32),
            device_float::rcp_rm_f32(a32),
            raw_float::rcp_rm_ftz_f32(a32),
            device_float::rcp_rp_f32(a32),
            raw_float::rcp_rp_ftz_f32(a32),
            device_float::rsqrt_approx_f32(a32),
            raw_float::rsqrt_approx_ftz_f32(a32),
            device_float::sqrt_approx_f32(a32),
            raw_float::sqrt_approx_ftz_f32(a32),
            device_float::sqrt_rn_f32(a32),
            raw_float::sqrt_rn_ftz_f32(a32),
            device_float::sqrt_rz_f32(a32),
            raw_float::sqrt_rz_ftz_f32(a32),
            device_float::sqrt_rm_f32(a32),
            raw_float::sqrt_rm_ftz_f32(a32),
            device_float::sqrt_rp_f32(a32),
            raw_float::sqrt_rp_ftz_f32(a32),
        ];
        let values_f64 = [
            raw_float::rcp_approx_ftz_f64(a64),
            device_float::rcp_rn_f64(a64),
            raw_float::rcp_rz_f64(a64),
            device_float::rcp_rm_f64(a64),
            raw_float::rcp_rp_f64(a64),
            device_float::rsqrt_approx_f64(a64),
            raw_float::sqrt_rn_f64(a64),
            device_float::sqrt_rz_f64(a64),
            raw_float::sqrt_rm_f64(a64),
            device_float::sqrt_rp_f64(a64),
        ];
        for (index, value) in values_f32.into_iter().enumerate() {
            if index < output_f32.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_f32.get_unchecked_mut(index) = value };
            }
        }
        for (index, value) in values_f64.into_iter().enumerate() {
            if index < output_f64.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_f64.get_unchecked_mut(index) = value };
            }
        }
    }

    /// Keeps every generated extended min/max intrinsic in the module.
    ///
    /// This coverage kernel is compiled but not launched by the example.
    #[kernel]
    pub fn compile_extended_minmax(
        mut output_f32: DisjointSlice<f32>,
        mut output_packed: DisjointSlice<u32>,
        a32: f32,
        b32: f32,
        a_packed: u32,
        b_packed: u32,
    ) {
        if cuda_device::thread::index_1d().get() != 0 {
            return;
        }
        let values_f32 = [
            raw_float::min_ftz_nan_xorsign_abs_f32(a32, b32),
            device_float::min_ftz_xorsign_abs_f32(a32, b32),
            raw_float::min_nan_xorsign_abs_f32(a32, b32),
            device_float::min_xorsign_abs_f32(a32, b32),
            raw_float::max_ftz_nan_xorsign_abs_f32(a32, b32),
            device_float::max_ftz_xorsign_abs_f32(a32, b32),
            raw_float::max_nan_xorsign_abs_f32(a32, b32),
            device_float::max_xorsign_abs_f32(a32, b32),
        ];
        let values_packed = [
            raw_f16x2::min_ftz_f16x2(a_packed, b_packed),
            device_f16x2::min_ftz_nan_f16x2(a_packed, b_packed),
            raw_f16x2::min_ftz_nan_xorsign_abs_f16x2(a_packed, b_packed),
            device_f16x2::min_ftz_xorsign_abs_f16x2(a_packed, b_packed),
            raw_bf16x2::min_nan_bf16x2(a_packed, b_packed),
            device_f16x2::min_nan_f16x2(a_packed, b_packed),
            device_bf16x2::min_nan_xorsign_abs_bf16x2(a_packed, b_packed),
            device_f16x2::min_nan_xorsign_abs_f16x2(a_packed, b_packed),
            raw_bf16x2::min_xorsign_abs_bf16x2(a_packed, b_packed),
            device_f16x2::min_xorsign_abs_f16x2(a_packed, b_packed),
            raw_f16x2::max_ftz_f16x2(a_packed, b_packed),
            device_f16x2::max_ftz_nan_f16x2(a_packed, b_packed),
            raw_f16x2::max_ftz_nan_xorsign_abs_f16x2(a_packed, b_packed),
            device_f16x2::max_ftz_xorsign_abs_f16x2(a_packed, b_packed),
            device_bf16x2::max_nan_bf16x2(a_packed, b_packed),
            device_f16x2::max_nan_f16x2(a_packed, b_packed),
            raw_bf16x2::max_nan_xorsign_abs_bf16x2(a_packed, b_packed),
            device_f16x2::max_nan_xorsign_abs_f16x2(a_packed, b_packed),
            device_bf16x2::max_xorsign_abs_bf16x2(a_packed, b_packed),
            device_f16x2::max_xorsign_abs_f16x2(a_packed, b_packed),
        ];
        for (index, value) in values_f32.into_iter().enumerate() {
            if index < output_f32.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_f32.get_unchecked_mut(index) = value };
            }
        }
        for (index, value) in values_packed.into_iter().enumerate() {
            if index < output_packed.len() {
                // SAFETY: the bounds check covers this unique output slot.
                unsafe { *output_packed.get_unchecked_mut(index) = value };
            }
        }
    }

    /// Keeps both generated movmatrix entry points in the compiled module.
    ///
    /// This coverage kernel is not launched. A caller must use one full warp.
    #[kernel]
    pub fn compile_movmatrix(mut output: DisjointSlice<u32>) {
        let lane = thread_idx_x() as usize;
        // SAFETY: every lane executes both calls in the same order.
        let value = unsafe {
            let raw = matrix::movmatrix_trans_b16(lane as u32);
            cuda_device::wmma::movmatrix_trans_b16(raw)
        };
        if lane < output.len() {
            // SAFETY: the bounds check covers this lane's unique slot.
            unsafe { *output.get_unchecked_mut(lane) = value };
        }
    }

    /// Keeps every generated byte-permutation mode in the compiled module.
    #[kernel]
    pub fn compile_prmt(mut output: DisjointSlice<u32>, a: u32, b: u32, control: u32) {
        let values = [
            prmt(a, b, control),
            prmt_f4e(a, b, control),
            prmt_b4e(a, b, control),
            prmt_rc8(a, control),
            prmt_ecl(a, control),
            prmt_ecr(a, control),
            prmt_rc16(a, control),
        ];
        let start = thread_idx_x() as usize * values.len();
        if start + values.len() <= output.len() {
            for (offset, value) in values.into_iter().enumerate() {
                // SAFETY: the bounds check covers this thread's unique slots.
                unsafe { *output.get_unchecked_mut(start + offset) = value };
            }
        }
    }

    /// Keeps every generated register-MMA variant in the compiled module.
    ///
    /// This coverage kernel is not launched by the example. A caller must use
    /// one complete warp because every MMA call is warp-synchronous.
    #[kernel]
    pub fn compile_register_mma(mut output: DisjointSlice<u64>) {
        // SAFETY: every lane executes the same instruction sequence. The zero
        // fragments have the documented register layouts.
        let (bf16, f16, tf32, f64) = unsafe {
            (
                matrix::mma_m16n8k16_f32_bf16([0.0; 4], [0; 4], [0; 2]),
                matrix::mma_m16n8k16_f32_f16([0.0; 4], [0; 4], [0; 2]),
                matrix::mma_m16n8k8_f32_tf32([0.0; 4], [0; 4], [0; 2]),
                matrix::mma_m8n8k4_f64([0.0; 2], 0.0, 0.0),
            )
        };
        let (tf32_k4, f16_k8, bf16_k8, f32_f16_k8, f16_k16) = unsafe {
            (
                matrix::mma_m16n8k4_f32_tf32([0.0; 4], [0; 2], 0),
                matrix::mma_m16n8k8_f16_f16([0; 2], [0; 2], 0),
                matrix::mma_m16n8k8_f32_bf16([0.0; 4], [0; 2], 0),
                matrix::mma_m16n8k8_f32_f16([0.0; 4], [0; 2], 0),
                matrix::mma_m16n8k16_f16_f16([0; 2], [0; 4], [0; 2]),
            )
        };
        // Keep the complete dense INT8 families in the generated path.
        let int8 = unsafe {
            [
                matrix::mma_m8n8k16_s32_s8([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_s8_u8([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_u8([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_u8_s8([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_s8_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_s8_u8_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_u8_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k16_s32_u8_s8_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m16n8k16_s32_s8([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_s8_u8([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_u8([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_u8_s8([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_s8_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_s8_u8_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_u8_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k16_s32_u8_s8_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_s8([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_s8_u8([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_u8([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_u8_s8([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_s8_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_s8_u8_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_u8_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k32_s32_u8_s8_satfinite([0; 4], [0; 4], [0; 2])[0],
            ]
        };
        // Keep the complete dense INT4 families in the generated path.
        let int4 = unsafe {
            [
                matrix::mma_m8n8k32_s32_s4([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_s4_u4([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_u4([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_u4_s4([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_s4_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_s4_u4_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_u4_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m8n8k32_s32_u4_s4_satfinite([0; 2], 0, 0)[0],
                matrix::mma_m16n8k32_s32_s4([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_s4_u4([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_u4([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_u4_s4([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_s4_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_s4_u4_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_u4_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k32_s32_u4_s4_satfinite([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k64_s32_s4([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_s4_u4([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_u4([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_u4_s4([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_s4_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_s4_u4_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_u4_satfinite([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m16n8k64_s32_u4_s4_satfinite([0; 4], [0; 4], [0; 2])[0],
            ]
        };
        // Keep every dense binary MMA form in the generated path.
        let b1 = unsafe {
            [
                matrix::mma_m8n8k128_s32_b1_xor_popc([0; 2], 0, 0)[0],
                matrix::mma_m16n8k128_s32_b1_xor_popc([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k256_s32_b1_xor_popc([0; 4], [0; 4], [0; 2])[0],
                matrix::mma_m8n8k128_s32_b1_and_popc([0; 2], 0, 0)[0],
                matrix::mma_m16n8k128_s32_b1_and_popc([0; 4], [0; 2], 0)[0],
                matrix::mma_m16n8k256_s32_b1_and_popc([0; 4], [0; 4], [0; 2])[0],
            ]
        };
        let standard_sparse_metadata = 0x1111_1111;
        // Keep every base sparse INT8 MMA form in the generated path.
        let sparse_int8 = unsafe {
            [
                matrix::mma_sp_m16n8k32_s32_s8([0; 4], [0; 2], [0; 2], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k32_s32_s8_u8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k32_s32_u8([0; 4], [0; 2], [0; 2], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k32_s32_u8_s8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k32_s32_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k32_s32_s8_u8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k32_s32_u8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k32_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
            ]
        };
        // Keep every base sparse m16n8k64 INT8 form in the generated path.
        let sparse_int8_k64 = unsafe {
            [
                matrix::mma_sp_m16n8k64_s32_s8([0; 4], [0; 4], [0; 4], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k64_s32_s8_u8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u8([0; 4], [0; 4], [0; 4], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k64_s32_u8_s8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_s8_u8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
            ]
        };
        // Keep every ordered-metadata sparse INT8 form in the generated path.
        let ordered_sparse_int8 = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_s8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_s8_u8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_u8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_u8_s8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_s8_u8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_u8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k32_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
            ]
        };
        // Keep every ordered-metadata m16n8k64 INT8 form in the generated path.
        let ordered_sparse_int8_k64 = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s8_u8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u8_s8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s8_u8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
            ]
        };
        // Keep every standard-metadata m16n8k64 INT4 form in the generated path.
        let sparse_int4_k64 = unsafe {
            [
                matrix::mma_sp_m16n8k64_s32_s4([0; 4], [0; 2], [0; 2], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k64_s32_s4_u4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u4([0; 4], [0; 2], [0; 2], standard_sparse_metadata, 0)
                    [0],
                matrix::mma_sp_m16n8k64_s32_u4_s4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k64_s32_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_s4_u4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k64_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
            ]
        };
        // Keep every standard-metadata m16n8k128 INT4 form in the generated path.
        let sparse_int4_k128 = unsafe {
            [
                matrix::mma_sp_m16n8k128_s32_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_s4_u4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_u4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_u4_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_s4_u4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_u4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                matrix::mma_sp_m16n8k128_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
            ]
        };
        // Keep every ordered-metadata m16n8k64 INT4 form in the generated path.
        let ordered_sparse_int4_k64 = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s4_u4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u4_s4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_s4_u4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k64_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
            ]
        };
        // Keep every ordered-metadata m16n8k128 INT4 form in the generated path.
        let ordered_sparse_int4_k128 = unsafe {
            [
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_s4_u4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_u4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_u4_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_s4_u4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_u4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                matrix::mma_sp_ordered_metadata_m16n8k128_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
            ]
        };
        // Compile representative stable cuda-device paths generated by the
        // same catalog. This kernel remains unlaunched.
        let compatibility = unsafe {
            [
                cuda_device::wmma::mma_m8n8k16_s32_s8_u8_satfinite([0; 2], 0, 0)[0],
                cuda_device::wmma::mma_m8n8k32_s32_s4_u4([0; 2], 0, 0)[0],
                cuda_device::wmma::mma_m8n8k32_s32_u4_s4_satfinite([0; 2], 0, 0)[0],
                cuda_device::wmma::mma_m16n8k16_s32_s8_u8([0; 4], [0; 2], 0)[0],
                cuda_device::wmma::mma_m16n8k32_s32_u8([0; 4], [0; 4], [0; 2])[0],
                cuda_device::wmma::mma_m16n8k32_s32_s4([0; 4], [0; 2], 0)[0],
                cuda_device::wmma::mma_m16n8k32_s32_u4([0; 4], [0; 2], 0)[0],
                cuda_device::wmma::mma_m16n8k64_s32_s4([0; 4], [0; 4], [0; 2])[0],
                cuda_device::wmma::mma_m16n8k64_s32_u4([0; 4], [0; 4], [0; 2])[0],
                cuda_device::wmma::mma_m8n8k128_s32_b1_xor_popc([0; 2], 0, 0)[0],
                cuda_device::wmma::mma_m16n8k128_s32_b1_xor_popc([0; 4], [0; 2], 0)[0],
                cuda_device::wmma::mma_m16n8k256_s32_b1_xor_popc([0; 4], [0; 4], [0; 2])[0],
                cuda_device::wmma::mma_m8n8k128_s32_b1_and_popc([0; 2], 0, 0)[0],
                cuda_device::wmma::mma_m16n8k128_s32_b1_and_popc([0; 4], [0; 2], 0)[0],
                cuda_device::wmma::mma_m16n8k256_s32_b1_and_popc([0; 4], [0; 4], [0; 2])[0],
                cuda_device::wmma::mma_sp_m16n8k32_s32_s8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k32_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k64_s32_s8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k64_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k64_s32_s4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k64_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    standard_sparse_metadata,
                    1,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k128_s32_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_m16n8k128_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    standard_sparse_metadata,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k32_s32_s8(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k32_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k64_s32_s8(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k64_s32_u8_s8_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k64_s32_s4(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k64_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 2],
                    [0; 2],
                    0x4444_4444,
                    1,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k128_s32_s4(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
                cuda_device::wmma::mma_sp_ordered_metadata_m16n8k128_s32_u4_s4_satfinite(
                    [0; 4],
                    [0; 4],
                    [0; 4],
                    0x4444_4444,
                    0,
                )[0],
            ]
        };
        let mut checksum = u64::from(bf16[0].to_bits())
            ^ u64::from(f16[0].to_bits())
            ^ u64::from(tf32[0].to_bits())
            ^ f64[0].to_bits()
            ^ u64::from(tf32_k4[0].to_bits())
            ^ u64::from(f16_k8[0])
            ^ u64::from(bf16_k8[0].to_bits())
            ^ u64::from(f32_f16_k8[0].to_bits())
            ^ u64::from(f16_k16[0]);
        for value in int8 {
            checksum ^= u64::from(value as u32);
        }
        for value in int4 {
            checksum ^= u64::from(value as u32);
        }
        for value in b1 {
            checksum ^= u64::from(value as u32);
        }
        for value in sparse_int8 {
            checksum ^= u64::from(value as u32);
        }
        for value in sparse_int8_k64 {
            checksum ^= u64::from(value as u32);
        }
        for value in ordered_sparse_int8 {
            checksum ^= u64::from(value as u32);
        }
        for value in ordered_sparse_int8_k64 {
            checksum ^= u64::from(value as u32);
        }
        for value in sparse_int4_k64 {
            checksum ^= u64::from(value as u32);
        }
        for value in sparse_int4_k128 {
            checksum ^= u64::from(value as u32);
        }
        for value in ordered_sparse_int4_k64 {
            checksum ^= u64::from(value as u32);
        }
        for value in ordered_sparse_int4_k128 {
            checksum ^= u64::from(value as u32);
        }
        for value in compatibility {
            checksum ^= u64::from(value as u32);
        }
        let index = thread_idx_x() as usize;
        if index < output.len() {
            // SAFETY: the bounds check covers this lane's unique slot.
            unsafe { *output.get_unchecked_mut(index) = checksum };
        }
    }

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
            sync_mask(member_mask);
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

        let down_delta = if lane < 31 { 1 } else { 0 };
        let up_delta = if lane > 0 { 1 } else { 0 };
        // SAFETY: both full warps execute the same shuffle sequence and mask.
        // Every computed source lane is active and named in `member_mask`.
        let (idx, bfly, down, up, idx_f32, bfly_f32, down_f32, up_f32) = unsafe {
            (
                shuffle_sync(member_mask, lane, 0),
                shuffle_xor_sync(member_mask, lane, 1),
                shuffle_down_sync(member_mask, lane, down_delta),
                shuffle_up_sync(member_mask, lane, up_delta),
                shuffle_f32_sync(member_mask, lane as f32, 0),
                shuffle_xor_f32_sync(member_mask, lane as f32, 1),
                shuffle_down_f32_sync(member_mask, lane as f32, down_delta),
                shuffle_up_f32_sync(member_mask, lane as f32, up_delta),
            )
        };
        let shuffles_ok = (idx == 0)
            & (bfly == (lane ^ 1))
            & (down == lane + down_delta)
            & (up == lane - up_delta)
            & (idx_f32 == 0.0)
            & (bfly_f32 == (lane ^ 1) as f32)
            & (down_f32 == (lane + down_delta) as f32)
            & (up_f32 == (lane - up_delta) as f32);

        // Distinct halves catch a split, source, or reassembly mistake.
        let wide_low_base = 0xa5a5_0000u64;
        let wide_value = ((lane as u64) << 32) | (wide_low_base + lane as u64);
        // SAFETY: both full warps execute the same shuffle sequence with the
        // full member mask. Every computed source lane is active and named.
        let (idx_u64, bfly_u64, down_u64, up_u64) = unsafe {
            (
                shuffle_u64_sync(member_mask, wide_value, 0),
                shuffle_xor_u64_sync(member_mask, wide_value, 1),
                shuffle_down_u64_sync(member_mask, wide_value, down_delta),
                shuffle_up_u64_sync(member_mask, wide_value, up_delta),
            )
        };
        let bfly_lane = lane ^ 1;
        let down_lane = lane + down_delta;
        let up_lane = lane - up_delta;
        let expected_bfly = ((bfly_lane as u64) << 32) | (wide_low_base + bfly_lane as u64);
        let expected_down = ((down_lane as u64) << 32) | (wide_low_base + down_lane as u64);
        let expected_up = ((up_lane as u64) << 32) | (wide_low_base + up_lane as u64);
        let shuffles_u64_ok = (idx_u64 == wide_low_base)
            & (bfly_u64 == expected_bfly)
            & (down_u64 == expected_down)
            & (up_u64 == expected_up);

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
            && shuffles_ok
            && shuffles_u64_ok
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
