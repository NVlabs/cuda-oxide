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
//! | Function                   | Bytes | Zero-fill | PTX                                                    |
//! |----------------------------|-------|-----------|--------------------------------------------------------|
//! | [`cp_async_ca_4`]          | 4     | No        | `cp.async.ca.shared.global [smem], [gmem], 4;`         |
//! | [`cp_async_ca_8`]          | 8     | No        | `cp.async.ca.shared.global [smem], [gmem], 8;`         |
//! | [`cp_async_ca_zfill_4`]    | 4     | Yes       | `cp.async.ca.shared.global [smem], [gmem], 4, src_sz;` |
//! | [`cp_async_ca_zfill_8`]    | 8     | Yes       | `cp.async.ca.shared.global [smem], [gmem], 8, src_sz;` |
//! | [`cp_async_ca_zfill_16`]   | 16    | Yes       | `cp.async.ca.shared.global [smem], [gmem],16, src_sz;` |
//!
//! # Notes
//!
//! The `.cg` (cache-global) cache policy is only supported for 16-byte copies,
//! so only `.ca` variants are provided here.
//!
//! The zero-fill variants copy `src_size` bytes from global memory and
//! zero-fill the remaining `cp_size - src_size` bytes in shared memory.
//! This is useful for boundary tiles in tiled algorithms where the last
//! tile may be smaller than the full tile size.
//! CUDA's shipped pipeline and CCCL helpers use `src_size == cp_size` for the
//! full-copy case, so the accepted range is `0..=cp_size`; larger values are
//! invalid.
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
/// `cp.async.ca.shared.global [shared_dst], [global_src], 4;`
///
/// # Safety
///
/// - `shared_dst` must point to 4 writable bytes in shared memory.
/// - `global_src` must point to 4 readable bytes in global memory.
/// - Both pointers must be aligned to 4 bytes.
/// - Both memory ranges must remain valid, and `global_src` must not be
///   modified, until the copy completes.
/// - `shared_dst` must not be read or written, including by an overlapping
///   asynchronous copy in the same group, until the copy completes.
/// - Before accessing the destination, the caller must complete this copy with
///   `cp.async.wait_all`, `cp.async.commit_group` followed by a matching
///   `cp.async.wait_group`, or an mbarrier that tracks this operation.
/// - Group waits cover copies issued by the executing thread. If another thread
///   will access the destination, synchronize the threads after completion.
/// - Completion instructions emitted with [`ptx_asm!`](crate::ptx_asm) must use
///   `clobber("memory")` so the compiler cannot move memory accesses across the
///   wait.
///
/// # See also
///
/// - [`cp_async_ca_8`]: 8-byte variant.
/// - [`cp_async_ca_zfill_4`]: 4-byte variant with zero-fill.
#[inline(never)]
pub unsafe fn cp_async_ca_4(_shared_dst: *mut u32, _global_src: *const u32) {
    // Lowered to inline PTX: cp.async.ca.shared.global [shared_dst], [global_src], 4;
    unreachable!("cp_async_ca_4 called outside CUDA kernel context")
}

/// Asynchronous 8-byte copy from global to shared memory with `.ca` cache policy.
///
/// Initiates an asynchronous copy of 8 bytes from global memory (`global_src`)
/// to shared memory (`shared_dst`) using the cache-all-levels (`.ca`) policy.
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [shared_dst], [global_src], 8;`
///
/// # Safety
///
/// - `shared_dst` must point to 8 writable bytes in shared memory.
/// - `global_src` must point to 8 readable bytes in global memory.
/// - Both pointers must be aligned to 8 bytes.
/// - Both memory ranges must remain valid, and `global_src` must not be
///   modified, until the copy completes.
/// - `shared_dst` must not be read or written, including by an overlapping
///   asynchronous copy in the same group, until the copy completes.
/// - Before accessing the destination, the caller must complete this copy with
///   `cp.async.wait_all`, `cp.async.commit_group` followed by a matching
///   `cp.async.wait_group`, or an mbarrier that tracks this operation.
/// - Group waits cover copies issued by the executing thread. If another thread
///   will access the destination, synchronize the threads after completion.
/// - Completion instructions emitted with [`ptx_asm!`](crate::ptx_asm) must use
///   `clobber("memory")` so the compiler cannot move memory accesses across the
///   wait.
///
/// # See also
///
/// - [`cp_async_ca_4`]: 4-byte variant.
/// - [`cp_async_ca_zfill_8`]: 8-byte variant with zero-fill.
#[inline(never)]
pub unsafe fn cp_async_ca_8(_shared_dst: *mut u32, _global_src: *const u32) {
    // Lowered to inline PTX: cp.async.ca.shared.global [shared_dst], [global_src], 8;
    unreachable!("cp_async_ca_8 called outside CUDA kernel context")
}

// =============================================================================
// cp.async with zero-fill (src_size parameter)
// =============================================================================

/// Asynchronous 4-byte copy from global to shared memory with zero-fill.
///
/// Copies `src_size` bytes from global memory to shared memory and
/// zero-fills the remaining `4 - src_size` bytes. When `src_size == 4`,
/// this behaves identically to [`cp_async_ca_4`].
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [shared_dst], [global_src], 4, src_size;`
///
/// # Safety
///
/// - `src_size` must be at most 4.
/// - `shared_dst` must point to 4 writable bytes in shared memory.
/// - `global_src` must be a global-memory pointer whose first `src_size` bytes
///   are readable.
/// - Both pointers must be aligned to 4 bytes.
/// - Both memory ranges must remain valid, and the readable prefix at
///   `global_src` must not be modified, until the copy completes.
/// - No operation may read or write any byte in the destination range,
///   including through an overlapping asynchronous copy, until this copy
///   completes.
/// - The thread issuing the copy must complete it with `cp.async.wait_all`,
///   `cp.async.commit_group` followed by a matching `cp.async.wait_group`, or an
///   mbarrier that tracks this operation. These waits cover only copies issued
///   by that thread.
/// - User-authored completion assembly must include a compiler memory clobber
///   (for example, `ptx_asm!(..., clobber("memory"))`) so memory accesses cannot
///   move across the completion wait.
/// - If another thread will access the destination, synchronize the threads
///   after the issuing thread has completed the copy.
///
/// # See also
///
/// - [`cp_async_ca_4`]: 4-byte copy without zero-fill.
/// - [`cp_async_ca_zfill_8`]: 8-byte variant.
/// - [`cp_async_ca_zfill_16`]: 16-byte variant.
#[inline(never)]
pub unsafe fn cp_async_ca_zfill_4(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32) {
    unreachable!("cp_async_ca_zfill_4 called outside CUDA kernel context")
}

/// Asynchronous 8-byte copy from global to shared memory with zero-fill.
///
/// Copies `src_size` bytes from global memory to shared memory and
/// zero-fills the remaining `8 - src_size` bytes. When `src_size == 8`,
/// this behaves identically to [`cp_async_ca_8`].
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [shared_dst], [global_src], 8, src_size;`
///
/// # Safety
///
/// - `src_size` must be at most 8.
/// - `shared_dst` must point to 8 writable bytes in shared memory.
/// - `global_src` must be a global-memory pointer whose first `src_size` bytes
///   are readable.
/// - Both pointers must be aligned to 8 bytes.
/// - Both memory ranges must remain valid, and the readable prefix at
///   `global_src` must not be modified, until the copy completes.
/// - No operation may read or write any byte in the destination range,
///   including through an overlapping asynchronous copy, until this copy
///   completes.
/// - The thread issuing the copy must complete it with `cp.async.wait_all`,
///   `cp.async.commit_group` followed by a matching `cp.async.wait_group`, or an
///   mbarrier that tracks this operation. These waits cover only copies issued
///   by that thread.
/// - User-authored completion assembly must include a compiler memory clobber
///   (for example, `ptx_asm!(..., clobber("memory"))`) so memory accesses cannot
///   move across the completion wait.
/// - If another thread will access the destination, synchronize the threads
///   after the issuing thread has completed the copy.
///
/// # See also
///
/// - [`cp_async_ca_8`]: 8-byte copy without zero-fill.
/// - [`cp_async_ca_zfill_4`]: 4-byte variant.
/// - [`cp_async_ca_zfill_16`]: 16-byte variant.
#[inline(never)]
pub unsafe fn cp_async_ca_zfill_8(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32) {
    unreachable!("cp_async_ca_zfill_8 called outside CUDA kernel context")
}

/// Asynchronous 16-byte copy from global to shared memory with zero-fill.
///
/// Copies `src_size` bytes from global memory to shared memory and
/// zero-fills the remaining `16 - src_size` bytes.
///
/// # PTX Instruction
///
/// `cp.async.ca.shared.global [shared_dst], [global_src], 16, src_size;`
///
/// # Safety
///
/// - `src_size` must be at most 16.
/// - `shared_dst` must point to 16 writable bytes in shared memory.
/// - `global_src` must be a global-memory pointer whose first `src_size` bytes
///   are readable.
/// - Both pointers must be aligned to 16 bytes.
/// - Both memory ranges must remain valid, and the readable prefix at
///   `global_src` must not be modified, until the copy completes.
/// - No operation may read or write any byte in the destination range,
///   including through an overlapping asynchronous copy, until this copy
///   completes.
/// - The thread issuing the copy must complete it with `cp.async.wait_all`,
///   `cp.async.commit_group` followed by a matching `cp.async.wait_group`, or an
///   mbarrier that tracks this operation. These waits cover only copies issued
///   by that thread.
/// - User-authored completion assembly must include a compiler memory clobber
///   (for example, `ptx_asm!(..., clobber("memory"))`) so memory accesses cannot
///   move across the completion wait.
/// - If another thread will access the destination, synchronize the threads
///   after the issuing thread has completed the copy.
///
/// # See also
///
/// - [`cp_async_ca_zfill_4`]: 4-byte variant.
/// - [`cp_async_ca_zfill_8`]: 8-byte variant.
#[inline(never)]
pub unsafe fn cp_async_ca_zfill_16(_shared_dst: *mut u32, _global_src: *const u8, _src_size: u32) {
    unreachable!("cp_async_ca_zfill_16 called outside CUDA kernel context")
}
