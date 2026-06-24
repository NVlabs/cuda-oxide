/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Asynchronous copy intrinsics (`cp.async`).
//!
//! These intrinsics perform asynchronous copies from global memory to shared
//! memory using the `.ca` (cache-all-levels) cache policy.
//!
//! # Variants
//!
//! | Function          | Bytes | Cache | PTX                                              |
//! |-------------------|-------|-------|--------------------------------------------------|
//! | [`cp_async_ca_4`] | 4     | `.ca` | `cp.async.ca.shared.global [smem], [gmem], 4;`   |
//! | [`cp_async_ca_8`] | 8     | `.ca` | `cp.async.ca.shared.global [smem], [gmem], 8;`   |
//!
//! # Notes
//!
//! The `.cg` (cache-global) cache policy is only supported for 16-byte copies,
//! so only `.ca` variants are provided for 4-byte and 8-byte copies.
//!
//! The functions are compiler-recognized stubs. Their bodies never execute; the
//! cuda-oxide compiler replaces each call with the corresponding PTX instruction.

/// Asynchronous 4-byte copy from global to shared memory with `.ca` cache policy.
///
/// Initiates an asynchronous copy of 4 bytes from global memory (`global_src`)
/// to shared memory (`shared_dst`) using the cache-all-levels (`.ca`) policy.
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [%smem32], [$1], 4;`
///
/// # Safety
///
/// - `shared_dst` must point to valid shared memory.
/// - `global_src` must point to valid global memory.
/// - Both pointers must be naturally aligned to 4 bytes.
/// - Must be followed by `cp.async.commit_group` and `cp.async.wait_group`.
///
/// # See also
///
/// - [`cp_async_ca_8`]: 8-byte variant.
#[inline(never)]
pub unsafe fn cp_async_ca_4(_shared_dst: *mut u32, _global_src: *const u32) {
    // Lowered to inline PTX: cp.async.ca.shared.global [%smem32], [$1], 4;
    unreachable!("cp_async_ca_4 called outside CUDA kernel context")
}

/// Asynchronous 8-byte copy from global to shared memory with `.ca` cache policy.
///
/// Initiates an asynchronous copy of 8 bytes from global memory (`global_src`)
/// to shared memory (`shared_dst`) using the cache-all-levels (`.ca`) policy.
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [%smem32], [$1], 8;`
///
/// # Safety
///
/// - `shared_dst` must point to valid shared memory.
/// - `global_src` must point to valid global memory.
/// - Both pointers must be naturally aligned to 4 bytes (the base alignment
///   requirement for `cp.async`).
/// - Must be followed by `cp.async.commit_group` and `cp.async.wait_group`.
///
/// # See also
///
/// - [`cp_async_ca_4`]: 4-byte variant.
#[inline(never)]
pub unsafe fn cp_async_ca_8(_shared_dst: *mut u32, _global_src: *const u32) {
    // Lowered to inline PTX: cp.async.ca.shared.global [%smem32], [$1], 8;
    unreachable!("cp_async_ca_8 called outside CUDA kernel context")
}
