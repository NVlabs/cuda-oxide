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

// =============================================================================
// Lane-Position Masks
// =============================================================================
//
// These five read-only special registers each return a 32-bit value whose
// bit `k` corresponds to lane `k` in the warp. They encode the calling lane's
// position relative to the rest of the warp and are the building blocks of
// warp-level scans, prefix sums, and stream compaction.
//
// A typical idiom combines a ballot with `lanemask_lt`:
//
// ```rust,ignore
// let active = warp::active_mask();
// let pred   = some_condition();
// let ballot = warp::ballot_sync(active, pred);
// // How many lanes *before* me also voted true → my output slot.
// let rank = (ballot & warp::lanemask_lt()).count_ones();
// ```
//
// Unlike the `*_sync` collectives these are plain register reads: they require
// no participation mask and are not warp-convergent.

/// Mask of all lanes with ID **strictly less** than the calling lane.
///
/// PTX `%lanemask_lt` (LLVM `@llvm.nvvm.read.ptx.sreg.lanemask.lt`). For lane
/// `i` the result is `(1 << i) - 1`. The canonical input to a warp prefix sum:
/// `(ballot & lanemask_lt()).count_ones()` is the number of earlier lanes that
/// satisfied the ballot predicate.
#[inline(never)]
pub fn lanemask_lt() -> u32 {
    // Recognized through the generated intrinsic catalog.
    unreachable!("lanemask_lt called outside CUDA kernel context")
}

/// Mask of all lanes with ID **less than or equal to** the calling lane.
///
/// PTX `%lanemask_le` (LLVM `@llvm.nvvm.read.ptx.sreg.lanemask.le`). For lane
/// `i` the result is `(1 << (i + 1)) - 1` (i.e. `lanemask_lt() | lanemask_eq()`),
/// giving an inclusive prefix mask.
#[inline(never)]
pub fn lanemask_le() -> u32 {
    // Recognized through the generated intrinsic catalog.
    unreachable!("lanemask_le called outside CUDA kernel context")
}

/// Mask with **only the calling lane's** bit set.
///
/// PTX `%lanemask_eq` (LLVM `@llvm.nvvm.read.ptx.sreg.lanemask.eq`). For lane
/// `i` the result is `1 << i` — equivalent to `1u32 << lane_id()` but read
/// directly from a hardware register.
#[inline(never)]
pub fn lanemask_eq() -> u32 {
    // Recognized through the generated intrinsic catalog.
    unreachable!("lanemask_eq called outside CUDA kernel context")
}

/// Mask of all lanes with ID **greater than or equal to** the calling lane.
///
/// PTX `%lanemask_ge` (LLVM `@llvm.nvvm.read.ptx.sreg.lanemask.ge`). For lane
/// `i` the result sets bits `i..=31` (i.e. `lanemask_gt() | lanemask_eq()`).
#[inline(never)]
pub fn lanemask_ge() -> u32 {
    // Recognized through the generated intrinsic catalog.
    unreachable!("lanemask_ge called outside CUDA kernel context")
}

/// Mask of all lanes with ID **strictly greater** than the calling lane.
///
/// PTX `%lanemask_gt` (LLVM `@llvm.nvvm.read.ptx.sreg.lanemask.gt`). For lane
/// `i` the result sets bits `(i + 1)..=31`. Useful for "lanes after me" suffix
/// scans and for finding the next active lane via `(ballot & lanemask_gt())`.
#[inline(never)]
pub fn lanemask_gt() -> u32 {
    // Recognized through the generated intrinsic catalog.
    unreachable!("lanemask_gt called outside CUDA kernel context")
}

include!("generated/warp_sreg.rs");

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
/// let mask = warp::ballot_sync(u32::MAX, some_predicate);
/// if some_predicate {
///     // Every lane in `mask` must reach this call.
///     warp::sync_mask(mask);
///     let leader = mask.trailing_zeros();
///     let value = warp::shuffle_sync(mask, my_value, leader);
/// }
/// ```
#[inline(never)]
pub fn sync_mask(mask: u32) {
    let _ = mask;
    unreachable!("sync_mask called outside CUDA kernel context")
}

/// Bitmask of lanes active at this instruction.
///
/// PTX `activemask.b32` (PTX 6.2+, sm_30+). Returns a 32-bit value where bit
/// `k` is set when lane `k` is active as this instruction executes.
///
/// This is only a snapshot. It does not prove that the returned lanes are
/// converged or will all execute a later collective. In straight-line,
/// full-warp code the result is normally `0xFFFFFFFF`.
///
/// # Common uses
///
/// - Inspect which lanes are active at a specific point.
/// - Capture the mask used by [`crate::cooperative_groups::CoalescedThreads`].
///
/// # Example
///
/// ```rust,ignore
/// if some_predicate {
///     // Only some lanes get here. Observe the active lanes at this point.
///     let mask = warp::active_mask();
///     let count = mask.count_ones();
///     let leader = mask.trailing_zeros();
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
// Warp Shuffle - 64-bit (u64 / f64)
// =============================================================================
//
// PTX `shfl.sync` only moves 32-bit registers — there is no `shfl.sync.*.b64`
// instruction and no `@llvm.nvvm.shfl.sync.*.i64` intrinsic. A 64-bit shuffle
// is therefore two 32-bit shuffles: split the value into its low/high halves,
// shuffle each with the same lane argument, and reassemble. We do that split in
// one compiler-visible convergent inline-PTX block
// (`mov.b64 {lo,hi}, x; shfl…; shfl…; mov.b64`). The hardware still executes
// two sequential b32 collectives; the block keeps the compiler from separating
// them.
//
// `u64` is the carrier (data movement is bit-exact, so it also covers `i64` —
// cast with `as u64` / `as i64`). The `f64` forms are zero-cost wrappers that
// bitcast through `u64`, mirroring how the 32-bit API offers `u32` and `f32`.

/// Shuffle (masked) u64: read `var` from `src_lane` for the given participation mask.
///
/// 64-bit analogue of [`shuffle_sync`] (PTX `shfl.sync.idx`, decomposed into two
/// `shfl.sync.idx.b32`). The full-warp shorthand is [`shuffle_u64`].
///
/// # Parameters
///
/// - `mask`: warp lane participation mask (`u32::MAX` = all 32 lanes)
/// - `var`: the 64-bit value to share (each lane provides its own)
/// - `src_lane`: the lane ID (0-31) to read from
///
/// # Participation contract
///
/// This compatibility function keeps its existing safe signature. The calling
/// lane must be named in `mask`, and every non-exited named lane must execute
/// the same shuffle with the same mask. A source lane computed as in range by
/// PTX must be active and named in `mask`; if PTX marks it out of range, the
/// calling lane's input is copied.
/// On `sm_6x` and earlier, all named lanes must execute in convergence, and no
/// lane outside `mask` may be active.
#[inline(never)]
pub fn shuffle_u64_sync(mask: u32, var: u64, src_lane: u32) -> u64 {
    let _ = (mask, var, src_lane);
    unreachable!("shuffle_u64_sync called outside CUDA kernel context")
}

/// Shuffle XOR (masked) u64: butterfly exchange under a mask.
///
/// 64-bit analogue of [`shuffle_xor_sync`] (PTX `shfl.sync.bfly`). The full-warp
/// shorthand is [`shuffle_xor_u64`].
///
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(never)]
pub fn shuffle_xor_u64_sync(mask: u32, var: u64, lane_mask: u32) -> u64 {
    let _ = (mask, var, lane_mask);
    unreachable!("shuffle_xor_u64_sync called outside CUDA kernel context")
}

/// Shuffle down (masked) u64: read from `(lane_id + delta)` under a mask.
///
/// 64-bit analogue of [`shuffle_down_sync`] (PTX `shfl.sync.down`). The full-warp
/// shorthand is [`shuffle_down_u64`].
///
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(never)]
pub fn shuffle_down_u64_sync(mask: u32, var: u64, delta: u32) -> u64 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_down_u64_sync called outside CUDA kernel context")
}

/// Shuffle up (masked) u64: read from `(lane_id - delta)` under a mask.
///
/// 64-bit analogue of [`shuffle_up_sync`] (PTX `shfl.sync.up`). The full-warp
/// shorthand is [`shuffle_up_u64`].
///
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(never)]
pub fn shuffle_up_u64_sync(mask: u32, var: u64, delta: u32) -> u64 {
    let _ = (mask, var, delta);
    unreachable!("shuffle_up_u64_sync called outside CUDA kernel context")
}

/// Shuffle u64 (full-warp): equivalent to [`shuffle_u64_sync`]`(u32::MAX, ...)`.
/// All 32 non-exited lanes must execute the same shuffle. A source computed as
/// in range by PTX must be active.
#[inline(always)]
pub fn shuffle_u64(var: u64, src_lane: u32) -> u64 {
    shuffle_u64_sync(u32::MAX, var, src_lane)
}

/// Shuffle XOR u64 (full-warp): equivalent to [`shuffle_xor_u64_sync`]`(u32::MAX, ...)`.
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(always)]
pub fn shuffle_xor_u64(var: u64, lane_mask: u32) -> u64 {
    shuffle_xor_u64_sync(u32::MAX, var, lane_mask)
}

/// Shuffle down u64 (full-warp): equivalent to [`shuffle_down_u64_sync`]`(u32::MAX, ...)`.
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(always)]
pub fn shuffle_down_u64(var: u64, delta: u32) -> u64 {
    shuffle_down_u64_sync(u32::MAX, var, delta)
}

/// Shuffle up u64 (full-warp): equivalent to [`shuffle_up_u64_sync`]`(u32::MAX, ...)`.
/// The participation and source requirements are the same as
/// [`shuffle_u64_sync`].
#[inline(always)]
pub fn shuffle_up_u64(var: u64, delta: u32) -> u64 {
    shuffle_up_u64_sync(u32::MAX, var, delta)
}

/// Shuffle (masked) f64: float variant of [`shuffle_u64_sync`].
///
/// Bitcasts through `u64` (`f64::to_bits` / `f64::from_bits`), so it moves the
/// exact bit pattern — NaN payloads are preserved.
#[inline(always)]
pub fn shuffle_f64_sync(mask: u32, var: f64, src_lane: u32) -> f64 {
    f64::from_bits(shuffle_u64_sync(mask, var.to_bits(), src_lane))
}

/// Shuffle XOR (masked) f64: float variant of [`shuffle_xor_u64_sync`].
#[inline(always)]
pub fn shuffle_xor_f64_sync(mask: u32, var: f64, lane_mask: u32) -> f64 {
    f64::from_bits(shuffle_xor_u64_sync(mask, var.to_bits(), lane_mask))
}

/// Shuffle down (masked) f64: float variant of [`shuffle_down_u64_sync`].
#[inline(always)]
pub fn shuffle_down_f64_sync(mask: u32, var: f64, delta: u32) -> f64 {
    f64::from_bits(shuffle_down_u64_sync(mask, var.to_bits(), delta))
}

/// Shuffle up (masked) f64: float variant of [`shuffle_up_u64_sync`].
#[inline(always)]
pub fn shuffle_up_f64_sync(mask: u32, var: f64, delta: u32) -> f64 {
    f64::from_bits(shuffle_up_u64_sync(mask, var.to_bits(), delta))
}

/// Shuffle f64 (full-warp): equivalent to [`shuffle_f64_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_f64(var: f64, src_lane: u32) -> f64 {
    shuffle_f64_sync(u32::MAX, var, src_lane)
}

/// Shuffle XOR f64 (full-warp): equivalent to [`shuffle_xor_f64_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_xor_f64(var: f64, lane_mask: u32) -> f64 {
    shuffle_xor_f64_sync(u32::MAX, var, lane_mask)
}

/// Shuffle down f64 (full-warp): equivalent to [`shuffle_down_f64_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_down_f64(var: f64, delta: u32) -> f64 {
    shuffle_down_f64_sync(u32::MAX, var, delta)
}

/// Shuffle up f64 (full-warp): equivalent to [`shuffle_up_f64_sync`]`(u32::MAX, ...)`.
#[inline(always)]
pub fn shuffle_up_f64(var: f64, delta: u32) -> f64 {
    shuffle_up_f64_sync(u32::MAX, var, delta)
}

// =============================================================================
// Warp Vote Operations
// =============================================================================

/// Vote ALL (masked): true if `predicate` holds for every participating lane.
///
/// PTX `vote.sync.all`. The full-warp shorthand is [`all`].
#[inline(never)]
pub fn all_sync(mask: u32, predicate: bool) -> bool {
    // Recognized through the generated intrinsic catalog.
    let _ = (mask, predicate);
    unreachable!("all_sync called outside CUDA kernel context")
}

/// Vote ANY (masked): true if `predicate` holds for at least one participating lane.
///
/// PTX `vote.sync.any`. The full-warp shorthand is [`any`].
#[inline(never)]
pub fn any_sync(mask: u32, predicate: bool) -> bool {
    // Recognized through the generated intrinsic catalog.
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
    // Recognized through the generated intrinsic catalog.
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

/// Match-all (32-bit, masked): participating-lane mask if every value agrees, else 0.
///
/// PTX `match.all.sync.b32`. Lowered to `@llvm.nvvm.match.all.sync.i32p`
/// with the predicate field discarded. Requires sm_70+.
///
/// Returns the non-exited participating lanes if every participating lane has
/// the same `value`; otherwise 0. Recover the all-match predicate as
/// `result != 0`.
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
/// Every non-exited lane named in `mask` must execute the same reduction
/// instruction, with the same qualifiers and `mask`; the calling lane must
/// itself be named in `mask`. The instruction waits for those lanes, so a
/// separate [`sync_mask`] is not required merely because they arrived through
/// divergent control flow. Violating the participation contract makes the PTX
/// behavior undefined. This runtime contract is distinct from LLVM's
/// `convergent` attribute, which constrains compiler motion and duplication.
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
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_min_u32(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_min_u32 called outside CUDA kernel context")
}

/// Warp-wide signed minimum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.min` → PTX `redux.sync.min.s32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_min_i32(mask: u32, value: i32) -> i32 {
    let _ = (mask, value);
    unreachable!("redux_sync_min_i32 called outside CUDA kernel context")
}

/// Warp-wide unsigned maximum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.umax` → PTX `redux.sync.max.u32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_max_u32(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_max_u32 called outside CUDA kernel context")
}

/// Warp-wide signed maximum (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.max` → PTX `redux.sync.max.s32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_max_i32(mask: u32, value: i32) -> i32 {
    let _ = (mask, value);
    unreachable!("redux_sync_max_i32 called outside CUDA kernel context")
}

/// Warp-wide bitwise AND reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.and` → PTX `redux.sync.and.b32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_and(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_and called outside CUDA kernel context")
}

/// Warp-wide bitwise OR reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.or` → PTX `redux.sync.or.b32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_or(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_or called outside CUDA kernel context")
}

/// Warp-wide bitwise XOR reduction (single instruction, sm_80+).
///
/// Lowered to `@llvm.nvvm.redux.sync.xor` → PTX `redux.sync.xor.b32`.
/// Convergent; see [`redux_sync_add`] for the participation contract.
#[inline(never)]
pub fn redux_sync_xor(mask: u32, value: u32) -> u32 {
    let _ = (mask, value);
    unreachable!("redux_sync_xor called outside CUDA kernel context")
}

// =============================================================================
// Leader election (sm_90+)
// =============================================================================
//
// `elect.sync` collectively chooses one leader lane from those named in
// `mask`. The choice is deterministic for the same mask. Every participating
// lane receives the leader's lane id and whether it is the leader. It replaces
// a multi-instruction election sequence such as
//
// ```rust,ignore
// let active = warp::active_mask();
// let leader = active.trailing_zeros();   // lowest set bit
// let is_leader = warp::lane_id() == leader;
// ```
//
// with one instruction. The classic use is warp-aggregated work: elect one
// lane to perform a single atomic / allocation / write on behalf of the warp.

/// Elect a single leader lane from the participating `mask` (sm_90+).
///
/// PTX `elect.sync d|p, membermask`, lowered through the generated route for
/// the selected backend. PTX does not promise which participating lane is
/// chosen, but the choice is deterministic for the same mask. Returns
/// `(leader_lane, is_elected)`:
///
/// - `leader_lane`: the elected lane id, returned to each participating lane.
/// - `is_elected`: `true` only for the calling lane if it is the leader.
///
/// Requires Hopper (sm_90+). Convergent: every lane named in `mask` must be
/// converged at the call (see [`redux_sync_add`] for the full convergence
/// contract — it is a runtime requirement on the caller, distinct from the
/// `convergent` attribute on the lowered intrinsic).
///
/// Most callers only need "am I the leader?"; reach for [`is_elected_sync`]
/// in that case and let the leader-id field fold away.
///
/// # Example: warp-aggregated counter
///
/// ```rust,ignore
/// // One lane per warp bumps a global counter and broadcasts the base index.
/// let (leader, elected) = warp::elect_sync(u32::MAX);
/// let base = if elected {
///     atomic_add(global_counter, 32)   // only the leader writes
/// } else {
///     0
/// };
/// // Share the leader's result with the rest of the warp.
/// let base = warp::shuffle_sync(u32::MAX, base, leader);
/// ```
#[inline(never)]
pub fn elect_sync(mask: u32) -> (u32, bool) {
    let _ = mask;
    unreachable!("elect_sync called outside CUDA kernel context")
}

/// Whether the calling lane is the elected leader of `mask` (sm_90+).
///
/// Convenience wrapper over [`elect_sync`] for the common "do this once per
/// warp" pattern; the elected leader-id field is discarded (and folds away in
/// codegen). See [`elect_sync`] for the hardware semantics and convergence
/// contract.
///
/// # Example
///
/// ```rust,ignore
/// if warp::is_elected_sync(u32::MAX) {
///     // Runs on exactly one lane of the warp.
/// }
/// ```
#[inline(always)]
pub fn is_elected_sync(mask: u32) -> bool {
    elect_sync(mask).1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_64_bit_shuffle_signatures_stay_stable() {
        let _: [fn(u32, u64, u32) -> u64; 4] = [
            shuffle_u64_sync,
            shuffle_xor_u64_sync,
            shuffle_down_u64_sync,
            shuffle_up_u64_sync,
        ];
        let _: [fn(u64, u32) -> u64; 4] = [
            shuffle_u64,
            shuffle_xor_u64,
            shuffle_down_u64,
            shuffle_up_u64,
        ];
        let _: [fn(u32, f64, u32) -> f64; 4] = [
            shuffle_f64_sync,
            shuffle_xor_f64_sync,
            shuffle_down_f64_sync,
            shuffle_up_f64_sync,
        ];
        let _: [fn(f64, u32) -> f64; 4] = [
            shuffle_f64,
            shuffle_xor_f64,
            shuffle_down_f64,
            shuffle_up_f64,
        ];
    }
}
