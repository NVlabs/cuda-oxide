/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! WMMA (Warp Matrix Multiply-Accumulate) device-side stubs.
//!
//! These functions are placeholders that get intercepted during MIR import
//! and replaced with the corresponding `dialect-nvvm` operations.

/// Signed int4 MMA: m16n8k32, accumulates into s32.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.s4.s4.s32`
///
/// # Safety
/// Must be called by all threads in a warp.
#[allow(unused_variables)]
pub unsafe fn mma_m16n8k32_s32_s4(acc: &mut [i32; 4], a: &[u32; 2], b: &u32) {
    let _ = (acc, a, b);
    unreachable!("device stub: lowered to mma.sync PTX by the compiler")
}

/// Unsigned int4 MMA: m16n8k32, accumulates into s32.
///
/// PTX: `mma.sync.aligned.m16n8k32.row.col.s32.u4.u4.s32`
///
/// # Safety
/// Must be called by all threads in a warp.
#[allow(unused_variables)]
pub unsafe fn mma_m16n8k32_s32_u4(acc: &mut [i32; 4], a: &[u32; 2], b: &u32) {
    let _ = (acc, a, b);
    unreachable!("device stub: lowered to mma.sync PTX by the compiler")
}

/// Signed int4 MMA: m16n8k64, accumulates into s32.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32`
///
/// # Safety
/// Must be called by all threads in a warp.
#[allow(unused_variables)]
pub unsafe fn mma_m16n8k64_s32_s4(acc: &mut [i32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("device stub: lowered to mma.sync PTX by the compiler")
}

/// Unsigned int4 MMA: m16n8k64, accumulates into s32.
///
/// PTX: `mma.sync.aligned.m16n8k64.row.col.s32.u4.u4.s32`
///
/// # Safety
/// Must be called by all threads in a warp.
#[allow(unused_variables)]
pub unsafe fn mma_m16n8k64_s32_u4(acc: &mut [i32; 4], a: &[u32; 4], b: &[u32; 2]) {
    let _ = (acc, a, b);
    unreachable!("device stub: lowered to mma.sync PTX by the compiler")
}
