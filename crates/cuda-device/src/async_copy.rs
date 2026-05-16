/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ampere async copy intrinsics (`cp.async`).
//!
//! These intrinsics provide asynchronous global→shared memory copies on SM 80+
//! (Ampere and later). Unlike regular loads which block the warp until data
//! arrives, `cp.async` copies bypass the register file and allow the warp to
//! continue executing compute instructions while the copy completes in the
//! background.
//!
//! # Cache Policy
//!
//! `cp_async_cg_16` uses the `.cg` (cache global) cache policy, which caches
//! only in L2 and bypasses L1. This is optimal for streaming access patterns
//! like GEMM tile loads where data is used once then discarded.
//!
//! The functions are compiler-recognized stubs. Their bodies never execute;
//! the cuda-oxide compiler replaces each call with the corresponding PTX.

/// Asynchronous 16-byte copy from global to shared memory.
///
/// Copies 16 bytes (128 bits) from `global_src` to `shared_dst` asynchronously.
/// The copy uses `.cg` cache policy (cache in L2 only, bypass L1).
///
/// Both pointers must be 16-byte aligned.
///
/// Maps to PTX: `cp.async.cg.shared.global [shared_dst], [global_src], 16;`
///
/// # Safety
///
/// - `shared_dst` must point to shared memory and be 16-byte aligned
/// - `global_src` must point to global memory and be 16-byte aligned
/// - The caller must call `cp_async_commit_group()` after all copies in a group
/// - The caller must call `cp_async_wait_group()` or `cp_async_wait_all()`
///   before reading the shared memory destination
#[inline(never)]
pub fn cp_async_cg_16(shared_dst: *mut u8, global_src: *const u8) {
    let _ = (shared_dst, global_src);
    unreachable!("cp_async_cg_16 called outside CUDA kernel context")
}

/// Asynchronous 16-byte copy from global to shared memory with `.ca` cache policy.
///
/// Copies 16 bytes (128 bits) from `global_src` to `shared_dst` asynchronously.
/// The copy uses `.ca` cache policy (cache at ALL levels: L1 + L2). This is
/// beneficial for data that will be reused across multiple warps or iterations,
/// such as small activation matrices in GEMM.
///
/// Both pointers must be 16-byte aligned.
///
/// Maps to PTX: `cp.async.ca.shared.global [shared_dst], [global_src], 16;`
///
/// # Safety
///
/// - `shared_dst` must point to shared memory and be 16-byte aligned
/// - `global_src` must point to global memory and be 16-byte aligned
/// - The caller must call `cp_async_commit_group()` after all copies in a group
/// - The caller must call `cp_async_wait_group()` or `cp_async_wait_all()`
///   before reading the shared memory destination
#[inline(never)]
pub fn cp_async_ca_16(shared_dst: *mut u8, global_src: *const u8) {
    let _ = (shared_dst, global_src);
    unreachable!("cp_async_ca_16 called outside CUDA kernel context")
}
