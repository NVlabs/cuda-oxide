/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Block-level collective primitives.
//!
//! These utilities combine warp-level shuffles with shared memory and
//! `sync_threads()` to perform reductions across an entire thread block.
//!
//! # Scratch array
//!
//! Each reduction needs one `f32` slot per warp. The scratch is taken as
//! `&mut SharedArray<f32, N, ALIGN>` rather than as a `*mut f32`, so the
//! capacity `N` travels with the pointer and every access inside these helpers
//! is bounded by it. Handing a raw pointer across this boundary erases the
//! length, which leaves the helper unable to tell whether it was given enough
//! room and forces its internal accesses to be unchecked.
//!
//! A block of `block_size` threads uses `block_size / 32` slots, so `N` must be
//! at least that. Anything beyond `N` is clamped rather than written out of
//! bounds, so an undersized array yields a reduction over the first `N` warps
//! instead of memory corruption.
//!
//! # Participation
//!
//! Every thread in the block must reach the call. These use `sync_threads()`
//! internally, so a thread that returns early leaves the rest waiting on a
//! barrier it will never reach.

use crate::shared::SharedArray;
use crate::thread;
use crate::warp;

/// Number of threads per warp on all currently supported architectures.
const WARP_SIZE: u32 = 32;

/// Reduce-sum a scalar f32 across all threads in the block.
///
/// Returns the sum in **lane 0 of warp 0 only**. All other threads get an
/// unspecified value.
///
/// # Safety
///
/// - Every thread in the block must call this function. It contains a
///   `sync_threads()`, so an early return elsewhere in the block deadlocks.
/// - `block_size` must equal the actual block size and be a multiple of 32.
/// - Each thread passes its own `&mut` borrow of the same shared array. That
///   mirrors [`crate::DisjointSlice`]: the borrow is per-thread, the backing
///   memory is shared, and disjointness comes from each warp writing only its
///   own slot.
///
/// Bounds are not a safety obligation here. Writes are guarded by the array's
/// capacity `N`, so passing an undersized array reduces over fewer warps rather
/// than writing out of bounds.
///
/// # Example
///
/// ```rust,ignore
/// // 256 threads is 8 warps, so 8 slots are needed.
/// static mut SMEM: SharedArray<f32, 8> = SharedArray::UNINIT;
///
/// // SAFETY: every thread in the block reaches this call.
/// let total = unsafe { block::reduce_sum_f32(my_value, &mut *core::ptr::addr_of_mut!(SMEM), 256) };
/// if thread::threadIdx_x() == 0 {
///     // `total` is the sum over all 256 threads
/// }
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn reduce_sum_f32<const N: usize, const ALIGN: usize>(
    val: f32,
    smem: &mut SharedArray<f32, N, ALIGN>,
    block_size: u32,
) -> f32 {
    block_reduce(val, smem, block_size, 0.0, warp::reduce_sum_f32)
}

/// Reduce-max a scalar f32 across all threads in the block.
///
/// Returns the maximum in **lane 0 of warp 0 only**. All other threads get an
/// unspecified value.
///
/// # Safety
///
/// Same obligations as [`reduce_sum_f32`].
///
/// # Example
///
/// ```rust,ignore
/// static mut SMEM: SharedArray<f32, 8> = SharedArray::UNINIT;
/// // SAFETY: every thread in the block reaches this call.
/// let absmax = unsafe { block::reduce_max_f32(v.abs(), &mut *core::ptr::addr_of_mut!(SMEM), 256) };
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn reduce_max_f32<const N: usize, const ALIGN: usize>(
    val: f32,
    smem: &mut SharedArray<f32, N, ALIGN>,
    block_size: u32,
) -> f32 {
    block_reduce(
        val,
        smem,
        block_size,
        f32::NEG_INFINITY,
        warp::reduce_max_f32,
    )
}

/// Reduce-min a scalar f32 across all threads in the block.
///
/// Returns the minimum in **lane 0 of warp 0 only**. All other threads get an
/// unspecified value.
///
/// # Safety
///
/// Same obligations as [`reduce_sum_f32`].
///
/// # Example
///
/// ```rust,ignore
/// static mut SMEM: SharedArray<f32, 8> = SharedArray::UNINIT;
/// // SAFETY: every thread in the block reaches this call.
/// let lo = unsafe { block::reduce_min_f32(v, &mut *core::ptr::addr_of_mut!(SMEM), 256) };
/// ```
#[must_use]
#[inline(always)]
pub unsafe fn reduce_min_f32<const N: usize, const ALIGN: usize>(
    val: f32,
    smem: &mut SharedArray<f32, N, ALIGN>,
    block_size: u32,
) -> f32 {
    block_reduce(val, smem, block_size, f32::INFINITY, warp::reduce_min_f32)
}

/// Shared body of the block reductions.
///
/// The three public entry points differ only in their identity element and the
/// warp reduction they build on, so the barrier placement and slot arithmetic -
/// the parts that are easy to get subtly wrong - exist once.
///
/// `warp_reduce` is one of the [`crate::warp`] reductions, reused rather than
/// reimplemented here so the butterfly shuffle pattern has a single definition.
/// It must leave the reduced value in every lane, which those do.
#[inline(always)]
fn block_reduce<const N: usize, const ALIGN: usize>(
    val: f32,
    smem: &mut SharedArray<f32, N, ALIGN>,
    block_size: u32,
    identity: f32,
    warp_reduce: impl Fn(f32) -> f32,
) -> f32 {
    let tid = thread::threadIdx_x();
    let lane = warp::lane_id();
    let warp_id = (tid / WARP_SIZE) as usize;
    // Clamped to the capacity the type guarantees, so an undersized array
    // cannot turn into an out-of-bounds access.
    let num_warps = ((block_size / WARP_SIZE) as usize).min(N);

    // Step 1: reduce within each warp.
    let warp_result = warp_reduce(val);

    // Step 2: lane 0 of each warp publishes its result to that warp's slot.
    if lane == 0 && warp_id < N {
        smem[warp_id] = warp_result;
    }
    thread::sync_threads();

    // Step 3: warp 0 reduces the per-warp results.
    if warp_id == 0 {
        let slot = tid as usize;
        let v = if slot < num_warps {
            smem[slot]
        } else {
            identity
        };
        warp_reduce(v)
    } else {
        identity
    }
}
