/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-level primitives.
//!
//! These operations enable fast data exchange within a warp (32 threads)
//! without explicit synchronization. Unlike shared memory operations, warp
//! shuffles use registers and require no barriers.
//!
//! # Performance
//!
//! | Operation | Shared Memory | Warp Shuffle |
//! |-----------|---------------|--------------|
//! | Latency | ~20 cycles | ~2 cycles |
//! | Synchronization | Requires `sync_threads()` | Implicit within warp |
//! | Scope | Block (up to 1024 threads) | Warp (32 threads) |
//!
//! # Example: Warp Reduction
//!
//! ```rust,ignore
//! use cuda_device::{kernel, thread, warp};
//!
//! #[kernel]
//! pub fn warp_reduce_sum(data: &[f32], mut out: DisjointSlice<f32>) {
//!     let gid = thread::index_1d();
//!     let lane = warp::lane_id();
//!
//!     let mut val = data[gid.get()];
//!
//!     // Butterfly reduction using shuffle_xor
//!     val = val + warp::shuffle_xor_f32(val, 16);
//!     val = val + warp::shuffle_xor_f32(val, 8);
//!     val = val + warp::shuffle_xor_f32(val, 4);
//!     val = val + warp::shuffle_xor_f32(val, 2);
//!     val = val + warp::shuffle_xor_f32(val, 1);
//!
//!     // Lane 0 has the sum
//!     if lane == 0 {
//!         let warp_idx = gid.get() / 32;
//!         *out.get_unchecked_mut(warp_idx) = val;
//!     }
//! }
//! ```

// =============================================================================
// Lane Identification
// =============================================================================

/// Get the lane ID within the current warp (0-31).
///
/// Each thread in a warp has a unique lane ID. This is useful for:
/// - Determining which thread should perform special actions (e.g., lane 0 writes output)
/// - Computing shuffle source lanes
/// - Implementing lane-specific logic
///
/// # Example
///
/// ```rust,ignore
/// let lane = warp::lane_id();
/// if lane == 0 {
///     // Only lane 0 writes the result
///     *output = result;
/// }
/// ```
#[inline(never)]
pub fn lane_id() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.laneid()
    unreachable!("lane_id called outside CUDA kernel context")
}

/// Synchronize a subset of warp lanes given by `mask`.
///
/// PTX `bar.warp.sync mask` (LLVM `@llvm.nvvm.bar.warp.sync(i32)`). All
/// lanes whose bit is set in `mask` must reach this call with the **same**
/// mask value before any of them proceeds. Lanes whose bit is clear are
/// not affected and need not reach the call.
///
/// This is the primitive that backs `CoalescedThreads::sync()` and
/// `WarpTile<N>::sync()` for sub-warp tiles. Straight-line warp-uniform
/// code does not need it — but on Volta and newer the SIMT reconvergence
/// model requires it after a divergent branch and before any other
/// `*.sync` collective on a subset of lanes.
///
/// # Example
///
/// ```rust,ignore
/// if some_predicate {
///     let mask = warp::active_mask();
///     // ... do divergent work ...
///     warp::sync_mask(mask);  // formal convergence point
///     let leader = mask.trailing_zeros();
///     let value = warp::shuffle_sync(mask, my_value, leader);
/// }
/// ```
#[inline(never)]
pub fn sync_mask(mask: u32) {
    let _ = mask;
    unreachable!("sync_mask called outside CUDA kernel context")
}

/// Bitmask of currently-converged lanes in this warp.
///
/// PTX `activemask.b32` (PTX 6.2+, sm_30+). Returns a 32-bit value where bit
/// `k` is set iff lane `k` is currently converged with this thread (i.e.
/// participating in this dynamic execution region).
///
/// In straight-line warp-uniform code this is `0xFFFFFFFF`. In divergent
/// branches it shrinks to the subset of lanes that took the same branch.
///
/// # Common uses
///
/// - **Build a mask for `*_sync` calls inside divergent code**: when only
///   some lanes reach a `ballot`/`shuffle`/`match` call site, pass
///   `active_mask()` as the mask so the intrinsic only synchronises the
///   participating lanes.
/// - **Construct a `CoalescedThreads` group**: the typed group's membership
///   set is the active mask captured at construction time.
///
/// # Example
///
/// ```rust,ignore
/// if some_predicate {
///     // Only some lanes get here. Build a mask of who's actually present.
///     let mask = warp::active_mask();
///     let count = mask.count_ones();        // how many lanes converged here
///     let leader = mask.trailing_zeros();   // lowest converged lane
/// }
/// ```
#[inline(never)]
pub fn active_mask() -> u32 {
    unreachable!("active_mask called outside CUDA kernel context")
}

/// Get the warp ID within the current block.
///
/// Computes: `threadIdx.x / 32`
///
/// This is a derived value, not a hardware register.
/// Only valid for 1D thread blocks; for multi-dimensional blocks,
/// compute your own warp ID from the linearized thread index.
#[inline(always)]
pub fn warp_id() -> u32 {
    crate::thread::threadIdx_x() / 32
}

// =============================================================================
// Masked sync intrinsics — operand convention
// =============================================================================
//
// The `*_sync(mask, ...)` functions below are the actual lowering targets.
// They take an explicit 32-bit warp participation mask: bit `k` set means
// lane `k` joins the collective. All non-exited lanes set in the mask must
// reach the call with the same mask value (PTX `*.sync` intrinsic
// constraints; see CUDA Programming Guide §5.4.6.6).
//
// The mask-less convenience functions (`ballot`, `shuffle`, ...) are
// `#[inline(always)]` wrappers that pass `u32::MAX` (full warp). After MIR
// inlining the codegen only ever sees the `*_sync` form.
//
// Typed group APIs (`WarpTile<N>`, `CoalescedThreads`) bake the right mask
// into the call site; they're built on top of these primitives.

// =============================================================================
// Warp Shuffle - Integer (u32)
// =============================================================================

/// Shuffle (masked): read `var` from `src_lane` for the given participation mask.
///
/// PTX `shfl.sync.idx.b32`. The full-warp shorthand is [`shuffle`].
///
/// # Parameters
///
/// - `mask`: warp lane participation mask (`u32::MAX` = all 32 lanes)
/// - `var`: the value to share (each lane provides its own)
/// - `src_lane`: the lane ID (0-31) to read from
#[inline(never)]
pub fn shuffle_sync(mask: u32, var: u32, src_lane: u32) -> u32 {
    let _ = (mask, var, src_lane);
    unreachable!("shuffle_sync called outside CUDA kernel context")
}

/// Shuffle XOR (masked): butterfly exchange under a mask.
///
/// PTX `shfl.sync.bfly.b32`. The full-warp shorthand is [`shuffle_xor`].
#[inline(never)]
pub fn shuffle_xor_sync(mask: u32, var: u32, lane_mask: u32) -> u32 {
    let _ = (mask, var, lane_mask);
    unreachable!("shuffle_xor_sync called outside CUDA kernel context")
}

/// Shuffle down (masked): read from `(lane_id + delta)` under a mask.
///
/// PTX `shfl.sync.down.b32`. The full-warp shorthand is [`shuffle_down`].
#[inline(never)]
pub fn shuffle_down_sync(mask: u32, var: u32, delta: u32) -> u32 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_down_sync called outside CUDA kernel context")
}

/// Shuffle up (masked): read from `(lane_id - delta)` under a mask.
///
/// PTX `shfl.sync.up.b32`. The full-warp shorthand is [`shuffle_up`].
#[inline(never)]
pub fn shuffle_up_sync(mask: u32, var: u32, delta: u32) -> u32 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_up_sync called outside CUDA kernel context")
}

/// Shuffle: get value from any lane in the warp (full-warp shorthand).
///
/// Equivalent to [`shuffle_sync`]`(u32::MAX, var, src_lane)`.
///
/// All 32 lanes of the warp must reach this call together. Use
/// [`shuffle_sync`] when you need to scope to a sub-warp.
///
/// # Example
///
/// ```rust,ignore
/// // Broadcast lane 0's value to all lanes
/// let broadcasted = warp::shuffle(my_value, 0);
/// ```
#[inline(always)]
pub fn shuffle(var: u32, src_lane: u32) -> u32 {
    shuffle_sync(u32::MAX, var, src_lane)
}

/// Shuffle XOR: butterfly exchange across the full warp.
///
/// Equivalent to [`shuffle_xor_sync`]`(u32::MAX, var, lane_mask)`.
///
/// # Example: Butterfly Reduction
///
/// ```rust,ignore
/// let mut sum = my_value;
/// sum = sum + warp::shuffle_xor(sum, 16);
/// sum = sum + warp::shuffle_xor(sum, 8);
/// sum = sum + warp::shuffle_xor(sum, 4);
/// sum = sum + warp::shuffle_xor(sum, 2);
/// sum = sum + warp::shuffle_xor(sum, 1);
/// ```
#[inline(always)]
pub fn shuffle_xor(var: u32, lane_mask: u32) -> u32 {
    shuffle_xor_sync(u32::MAX, var, lane_mask)
}

/// Shuffle down: read from `(lane_id + delta)` across the full warp.
///
/// Equivalent to [`shuffle_down_sync`]`(u32::MAX, var, delta)`.
#[inline(always)]
pub fn shuffle_down(var: u32, delta: u32) -> u32 {
    shuffle_down_sync(u32::MAX, var, delta)
}

/// Shuffle up: read from `(lane_id - delta)` across the full warp.
///
/// Equivalent to [`shuffle_up_sync`]`(u32::MAX, var, delta)`.
#[inline(always)]
pub fn shuffle_up(var: u32, delta: u32) -> u32 {
    shuffle_up_sync(u32::MAX, var, delta)
}

// =============================================================================
// Warp Shuffle - Float (f32)
// =============================================================================

/// Shuffle (masked) f32: float variant of [`shuffle_sync`].
#[inline(never)]
pub fn shuffle_f32_sync(mask: u32, var: f32, src_lane: u32) -> f32 {
    let _ = (mask, var, src_lane);
    unreachable!("shuffle_f32_sync called outside CUDA kernel context")
}

/// Shuffle XOR (masked) f32: float variant of [`shuffle_xor_sync`].
#[inline(never)]
pub fn shuffle_xor_f32_sync(mask: u32, var: f32, lane_mask: u32) -> f32 {
    let _ = (mask, var, lane_mask);
    unreachable!("shuffle_xor_f32_sync called outside CUDA kernel context")
}

/// Shuffle down (masked) f32: float variant of [`shuffle_down_sync`].
#[inline(never)]
pub fn shuffle_down_f32_sync(mask: u32, var: f32, delta: u32) -> f32 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_down_f32_sync called outside CUDA kernel context")
}

/// Shuffle up (masked) f32: float variant of [`shuffle_up_sync`].
#[inline(never)]
pub fn shuffle_up_f32_sync(mask: u32, var: f32, delta: u32) -> f32 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_up_f32_sync called outside CUDA kernel context")
}

/// Shuffle f32 (full-warp): equivalent to [`shuffle_f32_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_f32(var: f32, src_lane: u32) -> f32 {
    shuffle_f32_sync(u32::MAX, var, src_lane)
}

/// Shuffle XOR f32 (full-warp): equivalent to [`shuffle_xor_f32_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_xor_f32(var: f32, lane_mask: u32) -> f32 {
    shuffle_xor_f32_sync(u32::MAX, var, lane_mask)
}

/// Shuffle down f32 (full-warp): equivalent to [`shuffle_down_f32_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_down_f32(var: f32, delta: u32) -> f32 {
    shuffle_down_f32_sync(u32::MAX, var, delta)
}

/// Shuffle up f32 (full-warp): equivalent to [`shuffle_up_f32_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_up_f32(var: f32, delta: u32) -> f32 {
    shuffle_up_f32_sync(u32::MAX, var, delta)
}

// =============================================================================
// Warp Vote Operations
// =============================================================================

/// Vote ALL (masked): true if `predicate` holds for every participating lane.
///
/// PTX `vote.sync.all`. The full-warp shorthand is [`all`].
#[inline(never)]
pub fn all_sync(mask: u32, predicate: bool) -> bool {
    let _ = (mask, predicate);
    unreachable!("all_sync called outside CUDA kernel context")
}

/// Vote ANY (masked): true if `predicate` holds for at least one participating lane.
///
/// PTX `vote.sync.any`. The full-warp shorthand is [`any`].
#[inline(never)]
pub fn any_sync(mask: u32, predicate: bool) -> bool {
    let _ = (mask, predicate);
    unreachable!("any_sync called outside CUDA kernel context")
}

/// Vote BALLOT (masked): bitmask of lanes whose `predicate` is true.
///
/// PTX `vote.sync.ballot`. Returned bit `k` is set iff lane `k` is in `mask`
/// and its predicate is true; all other bits are 0. The full-warp shorthand
/// is [`ballot`].
#[inline(never)]
pub fn ballot_sync(mask: u32, predicate: bool) -> u32 {
    let _ = (mask, predicate);
    unreachable!("ballot_sync called outside CUDA kernel context")
}

/// Warp vote: returns true if ALL active threads have predicate true.
///
/// Equivalent to [`all_sync`]`(u32::MAX, predicate)`. Requires every lane
/// in the warp to reach the call.
///
/// # Example
///
/// ```rust,ignore
/// let all_valid = warp::all(my_value > 0.0);
/// ```
#[inline(always)]
pub fn all(predicate: bool) -> bool {
    all_sync(u32::MAX, predicate)
}

/// Warp vote: returns true if ANY active thread has predicate true.
///
/// Equivalent to [`any_sync`]`(u32::MAX, predicate)`.
///
/// # Example
///
/// ```rust,ignore
/// let any_overflow = warp::any(result > MAX_VALUE);
/// ```
#[inline(always)]
pub fn any(predicate: bool) -> bool {
    any_sync(u32::MAX, predicate)
}

/// Warp ballot: 32-bit mask where bit `i` indicates lane `i`'s predicate.
///
/// Equivalent to [`ballot_sync`]`(u32::MAX, predicate)`. Useful for counting
/// matching lanes, finding the first match, and implementing warp-level
/// control flow.
///
/// # Example
///
/// ```rust,ignore
/// let mask = warp::ballot(my_value > 0.0);
/// let count = mask.count_ones();
/// let first_positive_lane = mask.trailing_zeros();
/// ```
#[inline(always)]
pub fn ballot(predicate: bool) -> u32 {
    ballot_sync(u32::MAX, predicate)
}

/// Count threads with predicate true (population count of ballot).
///
/// Convenience function equivalent to `ballot(predicate).count_ones()`.
#[inline(always)]
pub fn popc(predicate: bool) -> u32 {
    ballot(predicate).count_ones()
}

// =============================================================================
// Warp Match Operations (sm_70+)
// =============================================================================
//
// `match.any.sync` and `match.all.sync` are warp-wide broadcast-and-compare
// instructions introduced on Volta. They take a 32-bit value from each
// participating lane and return a 32-bit bitmask describing which lanes share
// my value.
//
// Both come in 32-bit and 64-bit value variants, lowered to
// `@llvm.nvvm.match.{any,all}.sync.{i32,i64}` at codegen time.
//
// Use cases:
// - Bulk-insert deduplication: `match_any_sync(mask, key)` tells me which
//   lanes in the warp are inserting the *same* key, so the lowest such lane
//   can be the "winner" for the actual atomic write.
// - Cluster head detection: lane `k` is a cluster head iff bit `k` is the
//   lowest set bit in `match_any_sync(mask, value)`.
// - Equality reductions: `match_all_sync(mask, value) != 0` is true iff all
//   participating lanes hold the same value.
//
// Floating-point: bitcast the value to u32/u64 first. Cooperative groups
// match.any.sync compares bit patterns, so NaN handling is bit-exact (two
// NaNs match iff their bit representations match — the IEEE comparison
// semantics for NaN do *not* apply here).

/// Match-any (32-bit, masked): bitmask of lanes whose `value` equals mine.
///
/// PTX `match.any.sync.b32`. Lowered to `@llvm.nvvm.match.any.sync.i32`.
/// Requires sm_70+. Returned bit `k` is set iff lane `k` is in `mask` and
/// its `value` equals this lane's `value`.
///
/// # Example
///
/// ```rust,ignore
/// // Find the lowest lane in my warp that has my key (bulk-insert leader).
/// let same_key_lanes = warp::match_any_sync(u32::MAX, key);
/// let leader_lane = same_key_lanes.trailing_zeros();
/// if warp::lane_id() == leader_lane {
///     // I'm the leader for this key — do the atomic insert.
/// }
/// ```
#[inline(never)]
pub fn match_any_sync(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("match_any_sync called outside CUDA kernel context")
}

/// Match-any (64-bit value variant of [`match_any_sync`]).
///
/// PTX `match.any.sync.b64`. Lowered to `@llvm.nvvm.match.any.sync.i64`.
#[inline(never)]
pub fn match_any_i64_sync(mask: u32, value: u64) -> u32 {
    let _ = (mask, value);
    unreachable!("match_any_i64_sync called outside CUDA kernel context")
}

/// Match-all (32-bit, masked): full mask if every participating lane agrees, else 0.
///
/// PTX `match.all.sync.b32`. Lowered to `@llvm.nvvm.match.all.sync.i32p`
/// with the predicate field discarded. Requires sm_70+.
///
/// Returns `mask` if every lane in `mask` has the same `value`; otherwise 0.
/// Recover the all-match predicate as `result != 0`.
///
/// # Example
///
/// ```rust,ignore
/// if warp::match_all_sync(u32::MAX, my_value) != 0 {
///     // Every lane in the warp had the same value.
/// }
/// ```
#[inline(never)]
pub fn match_all_sync(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("match_all_sync called outside CUDA kernel context")
}

/// Match-all (64-bit value variant of [`match_all_sync`]).
///
/// PTX `match.all.sync.b64`. Lowered to `@llvm.nvvm.match.all.sync.i64p`.
#[inline(never)]
pub fn match_all_i64_sync(mask: u32, value: u64) -> u32 {
    let _ = (mask, value);
    unreachable!("match_all_i64_sync called outside CUDA kernel context")
}

/// Warp-wide sum reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.add` → PTX `redux.sync.add.s32`
/// (add is bit-identical for `s32`/`u32`, so this also covers `u32`).
/// Every lane named in `mask` contributes its `value`; the full sum is
/// broadcast back to all participating lanes. Convergent.
///
/// Works for both `u32` and `i32` addition (two's-complement wrap is
/// identical): to reduce an `i32`, call `redux_sync_add(mask, x as u32)`
/// and read the result back as `result as i32`.
///
/// # Convergence
///
/// Like all `*_sync` collectives, the lanes named in `mask` must be
/// **converged** at the call. Straight-line warp-uniform code is fine,
/// but after a divergent branch you must first reconverge the subset —
/// e.g. `warp::sync_mask(mask)` — otherwise the result is undefined.
/// (This is a runtime requirement on the caller; it is distinct from the
/// `convergent` attribute on the lowered intrinsic, which only stops LLVM
/// from moving the instruction across control flow.)
#[inline(never)]
pub fn redux_sync_add(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_add called outside CUDA kernel context")
}

// -----------------------------------------------------------------------------
// Integer min/max/and/or/xor reductions (sm_80+).
//
// Same shape and convergence rules as `redux_sync_add` (see its docs): every
// lane named in `mask` contributes its `value`, and the reduced result is
// broadcast back to all participating lanes.
//
// `min`/`max` come in signed (`_i32`) and unsigned (`_u32`) flavors because the
// comparison differs: e.g. `min(0xFFFFFFFF, 0)` is `-1` signed but `0` unsigned.
// `and`/`or`/`xor` are bitwise, so a single `u32` form covers `i32` too.
// -----------------------------------------------------------------------------

/// Warp-wide unsigned minimum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.umin` → PTX `redux.sync.min.u32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_min_u32(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_min_u32 called outside CUDA kernel context")
}

/// Warp-wide signed minimum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.min` → PTX `redux.sync.min.s32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_min_i32(mask: u32, value: i32) -> i32 {
    let _ = (mask, value);
    unreachable!("redux_sync_min_i32 called outside CUDA kernel context")
}

/// Warp-wide unsigned maximum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.umax` → PTX `redux.sync.max.u32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_max_u32(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_max_u32 called outside CUDA kernel context")
}

/// Warp-wide signed maximum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.max` → PTX `redux.sync.max.s32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_max_i32(mask: u32, value: i32) -> i32 {
    let _ = (mask, value);
    unreachable!("redux_sync_max_i32 called outside CUDA kernel context")
}

/// Warp-wide bitwise AND reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.and` → PTX `redux.sync.and.b32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_and(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_and called outside CUDA kernel context")
}

/// Warp-wide bitwise OR reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.or` → PTX `redux.sync.or.b32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_or(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_or called outside CUDA kernel context")
}

/// Warp-wide bitwise XOR reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.xor` → PTX `redux.sync.xor.b32`.
/// Convergent; participating lanes must be converged at the call.
#[inline(never)]
pub fn redux_sync_xor(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_xor called outside CUDA kernel context")
}
