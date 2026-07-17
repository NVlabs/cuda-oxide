/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Typed cooperative-groups handles.
//!
//! This module wraps the existing low-level intrinsics (`thread::*`,
//! `warp::*`, `cluster::*`, `grid::*`) in typed handles so the
//! participating-lane mask becomes part of the type rather than a silent
//! integer the caller has to remember.
//!
//! Each emitted PTX collective instruction (`vote.sync`, `shfl.sync`,
//! `match.sync`, ...) is byte-identical to the equivalent raw
//! `warp::*_sync` call — the participation mask folds at compile time.
//! Whether the wrapper *function* itself folds away is a separate
//! question, decided by rustc's MIR `Inline` cost threshold: for the
//! smallest wrappers it does and the call site is indistinguishable
//! from the raw form, but for larger wrappers the wrapper survives as
//! a `.visible .func` and the call site emits a `call.uni`. The
//! `hashmap_v2` example ships `find_kernel_warp` (raw) and
//! `find_kernel_warp_typed` (this API) side by side as an in-tree
//! reference for the current cost of that trade-off.
//!
//! # The universal trio
//!
//! Every group type implements [`ThreadGroup`] and exposes the same
//! three operations:
//!
//! - [`size`](ThreadGroup::size)         — how many threads are in this group
//! - [`thread_rank`](ThreadGroup::thread_rank) — my index within this group, `0..size`
//! - [`sync`](ThreadGroup::sync)         — barrier scoped to this group
//!
//! # The hierarchy
//!
//! ```text
//! Grid             this_grid()                grid::sync()             cooperative launch only
//! Cluster          this_cluster()             cluster::cluster_sync()  sm_90+
//! ThreadBlock      this_thread_block()        thread::sync_threads()
//!   WarpTile<32>     block.tiled_partition()  ballot/shfl/match        full warp
//!   WarpTile<N<32>>  block.tiled_partition()  warp::sync_mask          sub-warp tile
//!   CoalescedThreads coalesced_threads()      warp::sync_mask          runtime active mask
//! ```
//!
//! # Reductions and scans
//!
//! On top of the typed handles this module also ships the four
//! collective primitives `warp_reduce`, `warp_scan`, `block_reduce`,
//! and `block_scan`, generic over a [`ReduceOp`](ops::ReduceOp) trait
//! with concrete impls for `Sum`/`Min`/`Max` (`u32`/`i32`/`f32`) and
//! `BitAnd`/`BitOr`/`BitXor` (`u32` only). Block-scoped variants
//! take a `*mut SharedArray<T, NUM_WARPS>` for scratch and obey the
//! standard "block-sync between back-to-back uses of the same scratch"
//! contract documented on each function.
//!
//! # Example: warp-wide ballot from `find_kernel_warp`
//!
//! ```rust,ignore
//! use cuda_device::cooperative_groups::{this_thread_block, ThreadGroup, WarpCollective};
//!
//! let block = this_thread_block();
//! let warp = block.tiled_partition::<32>();
//!
//! let m_h2 = warp.ballot(tag == h2);
//! if m_h2 != 0 {
//!     let leader = m_h2.trailing_zeros();
//!     let payload = warp.shfl(my_payload, leader);
//!     // ...
//! }
//! ```
//!
//! Switching to a 16-lane sub-warp tile is one line:
//!
//! ```rust,ignore
//! let tile = block.tiled_partition::<16>();
//! let m_h2 = tile.ballot(tag == h2);  // mask is 0xFFFF or 0xFFFF0000
//! ```

use crate::{cluster, grid, shared::SharedArray, thread, warp};

// =============================================================================
// Traits
// =============================================================================

/// Universal "what every cooperative group exposes" trait.
///
/// Every group type — [`Grid`], [`Cluster`], [`ThreadBlock`],
/// [`WarpTile<N>`], [`CoalescedThreads`] — implements this. The three
/// methods are the entire reason cooperative groups exist as an
/// abstraction: at every level of the CUDA hierarchy you can ask "how
/// many of us are there", "which one am I", and "wait for everyone".
pub trait ThreadGroup: Copy {
    /// Number of threads in this group.
    fn size(&self) -> u32;

    /// This thread's index inside the group, `0..size()`.
    fn thread_rank(&self) -> u32;

    /// Barrier scoped to this group. Every thread in the group must reach
    /// this call before any of them proceeds.
    fn sync(&self);
}

/// Operations that only warp-scoped groups support.
///
/// Implemented by [`WarpTile<N>`] and [`CoalescedThreads`]. These are the
/// register-to-register collectives backed by PTX `shfl.sync` / `vote.sync`
/// / `match.sync` — they do not work at block, cluster, or grid scope.
///
/// All `*_rank` arguments and return values are **tile-relative**: lane 0
/// of a [`WarpTile<16>`] is the lane at warp position 0 *or* 16,
/// whichever bucket this thread is in. The implementation translates
/// to/from absolute warp lanes internally.
pub trait WarpCollective: ThreadGroup {
    /// Bitmask: bit `k` (within this group) set iff lane `k`'s `predicate` is true.
    ///
    /// PTX `vote.sync.ballot.b32` with the group's participation mask.
    fn ballot(&self, predicate: bool) -> u32;

    /// True iff every lane in this group has `predicate == true`.
    ///
    /// PTX `vote.sync.all.pred`.
    fn all(&self, predicate: bool) -> bool;

    /// True iff at least one lane in this group has `predicate == true`.
    ///
    /// PTX `vote.sync.any.pred`.
    fn any(&self, predicate: bool) -> bool;

    /// Read `var` from the lane at position `src_rank` in this group.
    ///
    /// PTX `shfl.sync.idx.b32`. `src_rank` is tile-relative (`0..size()`).
    fn shfl(&self, var: u32, src_rank: u32) -> u32;

    /// Butterfly exchange under this group's mask.
    ///
    /// PTX `shfl.sync.bfly.b32`. `lane_mask` is the XOR mask; for
    /// pairwise swaps of size `2^k` use `lane_mask = 1 << k`. A missing
    /// source in a sparse coalesced group returns the calling lane's value.
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32;

    /// Read from `(my_rank + delta)` within this group.
    ///
    /// PTX `shfl.sync.down.b32`. Lanes near the high end of the group
    /// receive their own value (no wraparound). Coalesced groups apply the
    /// delta to physical lanes and return self when that source is absent.
    fn shfl_down(&self, var: u32, delta: u32) -> u32;

    /// Read from `(my_rank - delta)` within this group.
    ///
    /// PTX `shfl.sync.up.b32`. Lanes near the low end of the group
    /// receive their own value (no wraparound). Coalesced groups apply the
    /// delta to physical lanes and return self when that source is absent.
    fn shfl_up(&self, var: u32, delta: u32) -> u32;

    /// `f32` variant of [`shfl`](Self::shfl).
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32;

    /// `f32` variant of [`shfl_xor`](Self::shfl_xor).
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32;

    /// `f32` variant of [`shfl_down`](Self::shfl_down).
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32;

    /// `f32` variant of [`shfl_up`](Self::shfl_up).
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32;

    /// Bitmask of lanes in this group whose `value` equals mine.
    ///
    /// PTX `match.any.sync.b32` (sm_70+). Bits are absolute warp-lane
    /// positions; the implementation already AND-s the raw hardware
    /// result with the group's participation mask, so the returned
    /// bits are guaranteed to be a subset of `mask()`.
    fn match_any(&self, value: u32) -> u32;

    /// 64-bit value variant of [`match_any`](Self::match_any).
    ///
    /// PTX `match.any.sync.b64` (sm_70+).
    fn match_any_i64(&self, value: u64) -> u32;

    /// Group's participation mask if every lane in the group has the same
    /// `value`, else 0.
    ///
    /// PTX `match.all.sync.b32` (sm_70+). Recover the all-match predicate
    /// as `result != 0`.
    fn match_all(&self, value: u32) -> u32;

    /// 64-bit value variant of [`match_all`](Self::match_all).
    fn match_all_i64(&self, value: u64) -> u32;
}

// =============================================================================
// Helpers (not part of the public surface)
// =============================================================================

/// Linearize the (x, y, z) thread index inside the current block.
#[inline(always)]
fn thread_in_block_linear() -> u32 {
    let tx = thread::threadIdx_x();
    let ty = thread::threadIdx_y();
    let tz = thread::threadIdx_z();
    let dx = thread::blockDim_x();
    let dy = thread::blockDim_y();
    (tz * dy + ty) * dx + tx
}

/// Total threads per block.
#[inline(always)]
fn threads_per_block() -> u32 {
    thread::blockDim_x() * thread::blockDim_y() * thread::blockDim_z()
}

/// Total blocks per grid.
#[inline(always)]
fn blocks_per_grid() -> u32 {
    thread::gridDim_x() * thread::gridDim_y() * thread::gridDim_z()
}

/// Linear block index inside the grid.
#[inline(always)]
fn block_in_grid_linear() -> u32 {
    let bx = thread::blockIdx_x();
    let by = thread::blockIdx_y();
    let bz = thread::blockIdx_z();
    let gx = thread::gridDim_x();
    let gy = thread::gridDim_y();
    (bz * gy + by) * gx + bx
}

// =============================================================================
// Grid
// =============================================================================

/// Handle for the entire grid: every thread in every block of the launch.
///
/// Construct with [`this_grid`]. Cooperative launches required for
/// [`sync`](ThreadGroup::sync) — see [`grid::sync`] for the contract.
#[derive(Copy, Clone, Debug)]
pub struct Grid {
    _priv: (),
}

/// Get the grid handle for this thread.
#[inline(always)]
pub fn this_grid() -> Grid {
    Grid { _priv: () }
}

impl Grid {
    /// Number of blocks in this grid.
    #[inline(always)]
    pub fn num_blocks(&self) -> u32 {
        blocks_per_grid()
    }

    /// This block's linear rank within the grid, `0 .. num_blocks()`.
    #[inline(always)]
    pub fn block_rank(&self) -> u32 {
        block_in_grid_linear()
    }
}

impl ThreadGroup for Grid {
    #[inline(always)]
    fn size(&self) -> u32 {
        blocks_per_grid() * threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        block_in_grid_linear() * threads_per_block() + thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        grid::sync();
    }
}

// =============================================================================
// Cluster
// =============================================================================

/// Handle for the current thread block cluster (sm_90+ only).
///
/// Construct with [`this_cluster`]. Outside a clustered launch every
/// query degenerates: `num_blocks() == 1`, `block_rank() == 0`, and
/// `sync()` is equivalent to a block barrier.
#[derive(Copy, Clone, Debug)]
pub struct Cluster {
    _priv: (),
}

/// Get the cluster handle for this thread.
#[inline(always)]
pub fn this_cluster() -> Cluster {
    Cluster { _priv: () }
}

impl Cluster {
    /// Number of blocks in this cluster.
    #[inline(always)]
    pub fn num_blocks(&self) -> u32 {
        cluster::cluster_size()
    }

    /// This block's linear rank within the cluster, `0 .. num_blocks()`.
    #[inline(always)]
    pub fn block_rank(&self) -> u32 {
        cluster::block_rank()
    }
}

impl ThreadGroup for Cluster {
    #[inline(always)]
    fn size(&self) -> u32 {
        cluster::cluster_size() * threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        cluster::block_rank() * threads_per_block() + thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        cluster::cluster_sync();
    }
}

// =============================================================================
// Thread block
// =============================================================================

/// Handle for the current thread block (CTA).
///
/// Construct with [`this_thread_block`]. This is the natural starting
/// point for partitioning: call [`tiled_partition`](ThreadBlock::tiled_partition)
/// to get a [`WarpTile<N>`] for the warp or sub-warp this thread belongs to.
#[derive(Copy, Clone, Debug)]
pub struct ThreadBlock {
    _priv: (),
}

/// Get the thread-block handle for this thread.
#[inline(always)]
pub fn this_thread_block() -> ThreadBlock {
    ThreadBlock { _priv: () }
}

impl ThreadBlock {
    /// Partition this block into tiles of `N` lanes.
    ///
    /// `N` must be a power of two in `1..=32`. For `N == 32` the result
    /// is a full-warp tile; for smaller `N` it's a sub-warp tile whose
    /// participation mask is computed at runtime from this lane's
    /// `lane_id()`.
    ///
    /// Compile-time `N` validation: a `const { assert!(...) }` block
    /// rejects illegal sizes at compile time.
    #[inline(always)]
    pub fn tiled_partition<const N: u32>(self) -> WarpTile<N> {
        const {
            assert!(
                N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32,
                "WarpTile size must be 1, 2, 4, 8, 16, or 32",
            );
        }
        WarpTile { _priv: () }
    }
}

impl ThreadGroup for ThreadBlock {
    #[inline(always)]
    fn size(&self) -> u32 {
        threads_per_block()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        thread_in_block_linear()
    }

    #[inline(always)]
    fn sync(&self) {
        thread::sync_threads();
    }
}

// =============================================================================
// WarpTile<N>
// =============================================================================

/// A tile of `N` lanes inside the current warp.
///
/// `N` is a compile-time constant, so the participation mask folds away
/// at codegen and the resulting PTX is byte-identical to a hand-written
/// `*_sync(mask, ...)` call.
///
/// Get one from [`ThreadBlock::tiled_partition`]:
///
/// ```rust,ignore
/// let warp = this_thread_block().tiled_partition::<32>();
/// let half = this_thread_block().tiled_partition::<16>();
/// ```
///
/// All [`WarpCollective`] methods take and return values relative to
/// **this tile**: `shfl(var, 0)` reads from the lowest lane of *this*
/// tile, not from absolute warp lane 0.
#[derive(Copy, Clone, Debug)]
pub struct WarpTile<const N: u32> {
    _priv: (),
}

impl<const N: u32> WarpTile<N> {
    /// Participation mask for this thread's tile.
    #[inline(always)]
    fn mask(&self) -> u32 {
        if N == 32 {
            u32::MAX
        } else {
            let lane = warp::lane_id();
            let tile_idx = lane / N;
            ((1u32 << N) - 1) << (tile_idx * N)
        }
    }

    /// Lane mask aligned to the start of this tile within the warp.
    #[inline(always)]
    fn tile_base_lane(&self) -> u32 {
        warp::lane_id() & !(N - 1)
    }

    #[inline(always)]
    fn xor_operand_or_self(&self, lane_mask: u32) -> u32 {
        if N == 32 || (self.thread_rank() ^ (lane_mask & 31)) < N {
            lane_mask
        } else {
            0
        }
    }

    #[inline(always)]
    fn down_operand_or_self(&self, delta: u32) -> u32 {
        if N == 32 || (delta & 31) < N - self.thread_rank() {
            delta
        } else {
            0
        }
    }

    #[inline(always)]
    fn up_operand_or_self(&self, delta: u32) -> u32 {
        if N == 32 || (delta & 31) <= self.thread_rank() {
            delta
        } else {
            0
        }
    }
}

impl<const N: u32> ThreadGroup for WarpTile<N> {
    #[inline(always)]
    fn size(&self) -> u32 {
        N
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        warp::lane_id() & (N - 1)
    }

    #[inline(always)]
    fn sync(&self) {
        if N == 32 {
            warp::sync_mask(u32::MAX);
        } else {
            warp::sync_mask(self.mask());
        }
    }
}

impl<const N: u32> WarpCollective for WarpTile<N> {
    #[inline(always)]
    fn ballot(&self, predicate: bool) -> u32 {
        let raw = warp::ballot_sync(self.mask(), predicate);
        if N == 32 {
            raw
        } else {
            (raw >> self.tile_base_lane()) & ((1u32 << N) - 1)
        }
    }

    #[inline(always)]
    fn all(&self, predicate: bool) -> bool {
        warp::all_sync(self.mask(), predicate)
    }

    #[inline(always)]
    fn any(&self, predicate: bool) -> bool {
        warp::any_sync(self.mask(), predicate)
    }

    #[inline(always)]
    fn shfl(&self, var: u32, src_rank: u32) -> u32 {
        let abs_src = self.tile_base_lane() | (src_rank & (N - 1));
        warp::shuffle_sync(self.mask(), var, abs_src)
    }

    #[inline(always)]
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32 {
        warp::shuffle_xor_sync(self.mask(), var, self.xor_operand_or_self(lane_mask))
    }

    #[inline(always)]
    fn shfl_down(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_down_sync(self.mask(), var, self.down_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_up(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_up_sync(self.mask(), var, self.up_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32 {
        let abs_src = self.tile_base_lane() | (src_rank & (N - 1));
        warp::shuffle_f32_sync(self.mask(), var, abs_src)
    }

    #[inline(always)]
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32 {
        warp::shuffle_xor_f32_sync(self.mask(), var, self.xor_operand_or_self(lane_mask))
    }

    #[inline(always)]
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_down_f32_sync(self.mask(), var, self.down_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_up_f32_sync(self.mask(), var, self.up_operand_or_self(delta))
    }

    #[inline(always)]
    fn match_any(&self, value: u32) -> u32 {
        warp::match_any_sync(self.mask(), value) & self.mask()
    }

    #[inline(always)]
    fn match_any_i64(&self, value: u64) -> u32 {
        warp::match_any_i64_sync(self.mask(), value) & self.mask()
    }

    #[inline(always)]
    fn match_all(&self, value: u32) -> u32 {
        warp::match_all_sync(self.mask(), value)
    }

    #[inline(always)]
    fn match_all_i64(&self, value: u64) -> u32 {
        warp::match_all_i64_sync(self.mask(), value)
    }
}

// =============================================================================
// CoalescedThreads
// =============================================================================

/// A snapshot of the warp lanes active when this group is created.
///
/// Construct with [`coalesced_threads`]. The participation mask is
/// captured from PTX `activemask.b32`. Every non-exited lane in that mask must
/// still execute each later group collective with the same mask.
///
/// For straight-line full-warp code, use [`WarpTile<32>`] instead.
#[derive(Copy, Clone, Debug)]
pub struct CoalescedThreads {
    mask: u32,
}

/// Capture the currently active lanes as a [`CoalescedThreads`] group.
#[inline(always)]
pub fn coalesced_threads() -> CoalescedThreads {
    CoalescedThreads {
        mask: warp::active_mask(),
    }
}

impl CoalescedThreads {
    /// The captured participation mask.
    #[inline(always)]
    pub fn raw_mask(&self) -> u32 {
        self.mask
    }

    #[inline(always)]
    fn xor_operand_or_self(&self, lane_mask: u32) -> u32 {
        let source = warp::lane_id() ^ (lane_mask & 31);
        if self.mask & (1u32 << source) != 0 {
            lane_mask
        } else {
            0
        }
    }

    #[inline(always)]
    fn down_operand_or_self(&self, delta: u32) -> u32 {
        let lane = warp::lane_id();
        let offset = delta & 31;
        if offset <= 31 - lane && self.mask & (1u32 << (lane + offset)) != 0 {
            delta
        } else {
            0
        }
    }

    #[inline(always)]
    fn up_operand_or_self(&self, delta: u32) -> u32 {
        let lane = warp::lane_id();
        let offset = delta & 31;
        if offset <= lane && self.mask & (1u32 << (lane - offset)) != 0 {
            delta
        } else {
            0
        }
    }
}

impl ThreadGroup for CoalescedThreads {
    #[inline(always)]
    fn size(&self) -> u32 {
        self.mask.count_ones()
    }

    #[inline(always)]
    fn thread_rank(&self) -> u32 {
        let lane = warp::lane_id();
        let lower = self.mask & ((1u32 << lane) - 1);
        lower.count_ones()
    }

    #[inline(always)]
    fn sync(&self) {
        warp::sync_mask(self.mask);
    }
}

impl WarpCollective for CoalescedThreads {
    #[inline(always)]
    fn ballot(&self, predicate: bool) -> u32 {
        warp::ballot_sync(self.mask, predicate) & self.mask
    }

    #[inline(always)]
    fn all(&self, predicate: bool) -> bool {
        warp::all_sync(self.mask, predicate)
    }

    #[inline(always)]
    fn any(&self, predicate: bool) -> bool {
        warp::any_sync(self.mask, predicate)
    }

    // `shfl` translates the group-relative `src_rank` to an absolute
    // warp lane (the lane whose own `thread_rank` inside the coalesced
    // group equals `src_rank`) by popping the `src_rank` lowest set
    // bits of the participation mask. When `src_rank` is a runtime
    // value the loop survives as a small `popc`/branch sequence; when
    // it folds to a constant the loop unrolls.
    //
    // The other shuffle modes use absolute warp lanes. When their computed
    // source is absent from this sparse group, the wrapper substitutes a
    // zero offset so the lane reads its own value.

    #[inline(always)]
    fn shfl(&self, var: u32, src_rank: u32) -> u32 {
        let mut m = self.mask;
        let mut k = src_rank;
        while k > 0 && m != 0 {
            m &= m - 1;
            k -= 1;
        }
        let abs_lane = if m == 0 {
            warp::lane_id()
        } else {
            m.trailing_zeros()
        };
        warp::shuffle_sync(self.mask, var, abs_lane)
    }

    #[inline(always)]
    fn shfl_xor(&self, var: u32, lane_mask: u32) -> u32 {
        warp::shuffle_xor_sync(self.mask, var, self.xor_operand_or_self(lane_mask))
    }

    #[inline(always)]
    fn shfl_down(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_down_sync(self.mask, var, self.down_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_up(&self, var: u32, delta: u32) -> u32 {
        warp::shuffle_up_sync(self.mask, var, self.up_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_f32(&self, var: f32, src_rank: u32) -> f32 {
        let mut m = self.mask;
        let mut k = src_rank;
        while k > 0 && m != 0 {
            m &= m - 1;
            k -= 1;
        }
        let abs_lane = if m == 0 {
            warp::lane_id()
        } else {
            m.trailing_zeros()
        };
        warp::shuffle_f32_sync(self.mask, var, abs_lane)
    }

    #[inline(always)]
    fn shfl_xor_f32(&self, var: f32, lane_mask: u32) -> f32 {
        warp::shuffle_xor_f32_sync(self.mask, var, self.xor_operand_or_self(lane_mask))
    }

    #[inline(always)]
    fn shfl_down_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_down_f32_sync(self.mask, var, self.down_operand_or_self(delta))
    }

    #[inline(always)]
    fn shfl_up_f32(&self, var: f32, delta: u32) -> f32 {
        warp::shuffle_up_f32_sync(self.mask, var, self.up_operand_or_self(delta))
    }

    #[inline(always)]
    fn match_any(&self, value: u32) -> u32 {
        warp::match_any_sync(self.mask, value) & self.mask
    }

    #[inline(always)]
    fn match_any_i64(&self, value: u64) -> u32 {
        warp::match_any_i64_sync(self.mask, value) & self.mask
    }

    #[inline(always)]
    fn match_all(&self, value: u32) -> u32 {
        warp::match_all_sync(self.mask, value)
    }

    #[inline(always)]
    fn match_all_i64(&self, value: u64) -> u32 {
        warp::match_all_i64_sync(self.mask, value)
    }
}

// =============================================================================
// Reductions and scans
// =============================================================================

/// Combiner operations for [`warp_reduce`], [`warp_scan`], [`block_reduce`],
/// and [`block_scan`].
///
/// Each combiner is a unit struct (`Sum`, `Min`, ...) that implements
/// `ReduceOp<T>` for one or more value types `T`. The unit lives in this
/// submodule so the names don't pollute the parent `cooperative_groups`
/// namespace; users typically write
/// `use cuda_device::cooperative_groups::ops::Sum;` at the call site.
///
/// Every `(combine, identity)` pair is a commutative associative monoid,
/// so the result is independent of the shuffle topology used to compute it.
pub mod ops {
    /// Binary reduction operation over values of type `T`.
    ///
    /// For determinism across the butterfly / Kogge-Stone topologies used
    /// by the reduce and scan primitives, `(combine, identity)` must form
    /// a commutative monoid. All operations in this module satisfy that.
    pub trait ReduceOp<T> {
        /// Identity element: `combine(identity(), x) == x` for all `x`.
        fn identity() -> T;
        /// Binary combiner. Must be associative and commutative.
        fn combine(a: T, b: T) -> T;
    }

    /// Sum: wrapping integer addition; IEEE-754 add for floats.
    #[derive(Copy, Clone, Debug)]
    pub struct Sum;

    /// Minimum (non-NaN-aware for floats — see the `f32` impl).
    #[derive(Copy, Clone, Debug)]
    pub struct Min;

    /// Maximum (non-NaN-aware for floats — see the `f32` impl).
    #[derive(Copy, Clone, Debug)]
    pub struct Max;

    /// Bitwise AND (`u32` only — meaningless for signed/float).
    #[derive(Copy, Clone, Debug)]
    pub struct BitAnd;

    /// Bitwise OR (`u32` only).
    #[derive(Copy, Clone, Debug)]
    pub struct BitOr;

    /// Bitwise XOR (`u32` only).
    #[derive(Copy, Clone, Debug)]
    pub struct BitXor;

    // --- u32 ---
    impl ReduceOp<u32> for Sum {
        #[inline(always)]
        fn identity() -> u32 {
            0
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            a.wrapping_add(b)
        }
    }
    impl ReduceOp<u32> for Min {
        #[inline(always)]
        fn identity() -> u32 {
            u32::MAX
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            if a < b { a } else { b }
        }
    }
    impl ReduceOp<u32> for Max {
        #[inline(always)]
        fn identity() -> u32 {
            0
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            if a > b { a } else { b }
        }
    }
    impl ReduceOp<u32> for BitAnd {
        #[inline(always)]
        fn identity() -> u32 {
            u32::MAX
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            a & b
        }
    }
    impl ReduceOp<u32> for BitOr {
        #[inline(always)]
        fn identity() -> u32 {
            0
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            a | b
        }
    }
    impl ReduceOp<u32> for BitXor {
        #[inline(always)]
        fn identity() -> u32 {
            0
        }
        #[inline(always)]
        fn combine(a: u32, b: u32) -> u32 {
            a ^ b
        }
    }

    // --- i32 ---
    impl ReduceOp<i32> for Sum {
        #[inline(always)]
        fn identity() -> i32 {
            0
        }
        #[inline(always)]
        fn combine(a: i32, b: i32) -> i32 {
            a.wrapping_add(b)
        }
    }
    impl ReduceOp<i32> for Min {
        #[inline(always)]
        fn identity() -> i32 {
            i32::MAX
        }
        #[inline(always)]
        fn combine(a: i32, b: i32) -> i32 {
            if a < b { a } else { b }
        }
    }
    impl ReduceOp<i32> for Max {
        #[inline(always)]
        fn identity() -> i32 {
            i32::MIN
        }
        #[inline(always)]
        fn combine(a: i32, b: i32) -> i32 {
            if a > b { a } else { b }
        }
    }

    // --- f32 ---
    impl ReduceOp<f32> for Sum {
        #[inline(always)]
        fn identity() -> f32 {
            0.0
        }
        #[inline(always)]
        fn combine(a: f32, b: f32) -> f32 {
            a + b
        }
    }
    impl ReduceOp<f32> for Min {
        #[inline(always)]
        fn identity() -> f32 {
            f32::INFINITY
        }
        // Direct comparison; NaN handling: `NaN < b` is false so a NaN in
        // `a` is dropped, but a NaN in `b` propagates. Inputs are expected
        // to be non-NaN; mixed-NaN reductions are a caller bug.
        #[inline(always)]
        fn combine(a: f32, b: f32) -> f32 {
            if a < b { a } else { b }
        }
    }
    impl ReduceOp<f32> for Max {
        #[inline(always)]
        fn identity() -> f32 {
            f32::NEG_INFINITY
        }
        #[inline(always)]
        fn combine(a: f32, b: f32) -> f32 {
            if a > b { a } else { b }
        }
    }
}

/// Scalars that ride a `WarpCollective` shuffle.
///
/// Reductions and scans need to shuffle the value being combined, but the
/// underlying PTX `shfl.sync.{bfly,up,down,idx}` only ships in `b32` and
/// `f32` flavors. This trait routes each supported `T` (`u32`, `i32`,
/// `f32`) to the right one and bitcasts where required (`i32`).
pub trait WarpShuffle: Copy + Sized {
    /// Butterfly exchange under the group's mask. See
    /// [`WarpCollective::shfl_xor`].
    fn shfl_xor_via<G: WarpCollective>(group: &G, value: Self, lane_mask: u32) -> Self;

    /// Read from `(my_rank - delta)` within the group. See
    /// [`WarpCollective::shfl_up`].
    fn shfl_up_via<G: WarpCollective>(group: &G, value: Self, delta: u32) -> Self;
}

impl WarpShuffle for u32 {
    #[inline(always)]
    fn shfl_xor_via<G: WarpCollective>(group: &G, value: Self, lane_mask: u32) -> Self {
        group.shfl_xor(value, lane_mask)
    }
    #[inline(always)]
    fn shfl_up_via<G: WarpCollective>(group: &G, value: Self, delta: u32) -> Self {
        group.shfl_up(value, delta)
    }
}

impl WarpShuffle for i32 {
    // `as u32` / `as i32` are bit-preserving in two's-complement; the b32
    // shuffle sees the same 32 bits, so the round-trip is lossless.
    #[inline(always)]
    fn shfl_xor_via<G: WarpCollective>(group: &G, value: Self, lane_mask: u32) -> Self {
        group.shfl_xor(value as u32, lane_mask) as i32
    }
    #[inline(always)]
    fn shfl_up_via<G: WarpCollective>(group: &G, value: Self, delta: u32) -> Self {
        group.shfl_up(value as u32, delta) as i32
    }
}

impl WarpShuffle for f32 {
    #[inline(always)]
    fn shfl_xor_via<G: WarpCollective>(group: &G, value: Self, lane_mask: u32) -> Self {
        group.shfl_xor_f32(value, lane_mask)
    }
    #[inline(always)]
    fn shfl_up_via<G: WarpCollective>(group: &G, value: Self, delta: u32) -> Self {
        group.shfl_up_f32(value, delta)
    }
}

/// Reduce one value per lane down to a single value across a `WarpTile<N>`.
///
/// Implements the standard butterfly reduction via `shfl.sync.bfly` —
/// `log2(N)` exchanges, each XOR-ing the lane id by `delta` and combining.
/// **Every lane in the tile receives the reduced value** (not just lane 0):
/// the butterfly converges all lanes onto the same answer, which is the
/// usual ergonomic shape and matches CUB's `WarpReduce`.
///
/// `N` is the tile size (1, 2, 4, 8, 16, or 32) — already validated by
/// [`ThreadBlock::tiled_partition`] at construction time.
///
/// # Convergence
///
/// All lanes named in the tile's participation mask must reach this call.
/// For sub-warp tiles the wrapper replaces an out-of-tile source with the
/// calling lane, so two tiles inside the same warp remain independent on
/// `sm_70` and newer. Older targets also require all active lanes to use the
/// same member mask.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::cooperative_groups::{this_thread_block, warp_reduce, ops::Sum};
///
/// let warp = this_thread_block().tiled_partition::<32>();
/// let total: u32 = warp_reduce::<u32, Sum, _>(&warp, my_value);
/// // every lane now holds the warp-wide sum
/// ```
#[inline(always)]
pub fn warp_reduce<T, Op, const N: u32>(tile: &WarpTile<N>, value: T) -> T
where
    T: WarpShuffle,
    Op: ops::ReduceOp<T>,
{
    let mut acc = value;
    let mut delta: u32 = N >> 1;
    while delta > 0 {
        let other = T::shfl_xor_via(tile, acc, delta);
        acc = Op::combine(acc, other);
        delta >>= 1;
    }
    acc
}

/// Inclusive scan across a `WarpTile<N>`. Each lane receives the reduction
/// of values from tile-rank `0..=my_rank`.
///
/// Implemented as a `log2(N)`-step Kogge-Stone scan: at step `delta` each
/// lane reads the value from rank `(my_rank - delta)` via `shfl.sync.up`,
/// then combines if `my_rank >= delta`. The wrapper replaces a cross-tile
/// source with the calling lane, so sub-warp tiles scan independently on
/// `sm_70` and newer. Older targets also require all active lanes to use the
/// same member mask.
///
/// `N` is the tile size (1, 2, 4, 8, 16, or 32).
///
/// # Example
///
/// ```rust,ignore
/// let warp = this_thread_block().tiled_partition::<32>();
/// let prefix: u32 = warp_scan::<u32, Sum, _>(&warp, 1u32);
/// // lane k now holds k + 1 (inclusive prefix of all-ones)
/// ```
#[inline(always)]
pub fn warp_scan<T, Op, const N: u32>(tile: &WarpTile<N>, value: T) -> T
where
    T: WarpShuffle,
    Op: ops::ReduceOp<T>,
{
    let mut acc = value;
    let rank = tile.thread_rank();
    let mut delta: u32 = 1;
    while delta < N {
        let other = T::shfl_up_via(tile, acc, delta);
        if rank >= delta {
            acc = Op::combine(acc, other);
        }
        delta <<= 1;
    }
    acc
}

/// Linear warp index inside the current block. Works for any block dim
/// shape; `warp::warp_id` only handles the 1D case.
#[inline(always)]
fn warp_in_block_linear() -> u32 {
    thread_in_block_linear() / 32
}

/// Reduce one value per thread down to a single value across a thread block.
///
/// Three-phase shape (the same shape CUB and cuCollections use):
///
/// 1. Each warp warp-reduces its 32 lanes' values via [`warp_reduce`].
/// 2. Lane 0 of each warp writes its warp's total to `smem[warp_id]`,
///    then `block.sync()`.
/// 3. The first warp loads the per-warp totals (filling unused slots with
///    `Op::identity()`), warp-reduces them, and lane 0 writes the
///    block-wide result back to `smem[0]`. After a second `block.sync()`
///    every thread reads `smem[0]`.
///
/// `NUM_WARPS` must equal `BLOCK_SIZE / 32`. It's a `usize` const-generic
/// so it lines up with the [`SharedArray`] length parameter; the
/// compile-time check rejects values outside `1..=32` (i.e. block sizes
/// outside `32..=1024`).
///
/// # Smem reuse
///
/// `smem` is left in an arbitrary state on return. Callers that reuse
/// the same `SharedArray` for a second reduction or scan in the same
/// kernel **must** place a `block.sync()` between calls — otherwise
/// late readers from the first call may race the writes of the second.
///
/// # Why a raw pointer?
///
/// We take `*mut SharedArray<T, NUM_WARPS>` rather than `&mut` so that
/// callers can pass `&raw mut SMEM` directly without an `unsafe` block
/// at the call site (Rust 2024 forbids `&mut` references to a
/// `static mut`, but `&raw mut` is safe). The unsafety — that nothing
/// else aliases the static for the duration of the call — is local to
/// this function body, where the helpers and barriers below preserve it.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::cooperative_groups::{block_reduce, ops::Sum, this_thread_block};
/// use cuda_device::SharedArray;
///
/// #[kernel]
/// pub fn my_kernel(...) {
///     // 8 = 256 / 32 warps per 256-thread block
///     static mut SMEM: SharedArray<u32, 8> = SharedArray::UNINIT;
///     let block = this_thread_block();
///     let total = block_reduce::<u32, Sum, _>(&block, my_value, &raw mut SMEM);
///     // every thread now holds the block-wide total
/// }
/// ```
#[inline(always)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn block_reduce<T, Op, const NUM_WARPS: usize>(
    block: &ThreadBlock,
    value: T,
    smem: *mut SharedArray<T, NUM_WARPS>,
) -> T
where
    T: WarpShuffle,
    Op: ops::ReduceOp<T>,
{
    const {
        assert!(
            NUM_WARPS > 0 && NUM_WARPS <= 32,
            "block_reduce requires 1..=32 warps (block size 32..=1024)",
        );
    }

    // SAFETY: The caller (a kernel) hands us a raw pointer to a `static mut
    // SharedArray<T, NUM_WARPS>`. Within this function:
    //   - all writes occur at distinct (warp_id, lane==0) or (lane==0) sites,
    //   - all reads are separated from prior writes by `block.sync()`.
    // No other reference to the static exists for the duration of the call.
    let smem: &mut SharedArray<T, NUM_WARPS> = unsafe { &mut *smem };

    let warp = block.tiled_partition::<32>();
    let lane = warp.thread_rank();
    let warp_id = warp_in_block_linear();

    let warp_total = warp_reduce::<T, Op, 32>(&warp, value);

    if lane == 0 {
        smem[warp_id as usize] = warp_total;
    }
    block.sync();

    if warp_id == 0 {
        let v: T = if (lane as usize) < NUM_WARPS {
            smem[lane as usize]
        } else {
            <Op as ops::ReduceOp<T>>::identity()
        };
        let block_total = warp_reduce::<T, Op, 32>(&warp, v);
        if lane == 0 {
            smem[0] = block_total;
        }
    }
    block.sync();

    smem[0]
}

/// Inclusive scan across a thread block. Thread `i` (in linear order
/// `(z * blockDim.y + y) * blockDim.x + x`) receives the reduction of
/// values from threads `0..=i`.
///
/// Three-phase shape:
///
/// 1. Each warp does an inclusive [`warp_scan`].
/// 2. Lane 31 of each warp writes its warp total (= last value of its
///    inclusive scan) to `smem[warp_id]`, then `block.sync()`.
/// 3. Warp 0 inclusive-scans the warp totals, then converts to an
///    exclusive prefix and writes back to `smem[0..NUM_WARPS]`. After
///    `block.sync()` every thread reads its own warp's exclusive prefix
///    and combines it with its intra-warp inclusive value.
///
/// `NUM_WARPS` must equal `BLOCK_SIZE / 32`. Same `SharedArray` reuse
/// contract as [`block_reduce`] — caller must `block.sync()` before
/// reusing the same smem. Same raw-pointer rationale as well: pass
/// `&raw mut SMEM` from the call site (no `unsafe` block needed).
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::cooperative_groups::{block_scan, ops::Sum, this_thread_block};
/// use cuda_device::SharedArray;
///
/// #[kernel]
/// pub fn my_kernel(...) {
///     static mut SMEM: SharedArray<u32, 8> = SharedArray::UNINIT;
///     let block = this_thread_block();
///     let prefix = block_scan::<u32, Sum, _>(&block, 1u32, &raw mut SMEM);
///     // thread i now holds i + 1 (inclusive scan of all-ones)
/// }
/// ```
#[inline(always)]
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn block_scan<T, Op, const NUM_WARPS: usize>(
    block: &ThreadBlock,
    value: T,
    smem: *mut SharedArray<T, NUM_WARPS>,
) -> T
where
    T: WarpShuffle,
    Op: ops::ReduceOp<T>,
{
    const {
        assert!(
            NUM_WARPS > 0 && NUM_WARPS <= 32,
            "block_scan requires 1..=32 warps (block size 32..=1024)",
        );
    }

    // SAFETY: see `block_reduce`.
    let smem: &mut SharedArray<T, NUM_WARPS> = unsafe { &mut *smem };

    let warp = block.tiled_partition::<32>();
    let lane = warp.thread_rank();
    let warp_id = warp_in_block_linear();

    let warp_inclusive = warp_scan::<T, Op, 32>(&warp, value);

    if lane == 31 {
        smem[warp_id as usize] = warp_inclusive;
    }
    block.sync();

    if warp_id == 0 {
        let v: T = if (lane as usize) < NUM_WARPS {
            smem[lane as usize]
        } else {
            <Op as ops::ReduceOp<T>>::identity()
        };
        let inclusive = warp_scan::<T, Op, 32>(&warp, v);
        // Inclusive -> exclusive: shift right by one lane and force
        // identity at lane 0. shfl_up at lane 0 returns own value, so
        // explicit branch is needed.
        let shifted = T::shfl_up_via(&warp, inclusive, 1);
        let exclusive = if lane == 0 {
            <Op as ops::ReduceOp<T>>::identity()
        } else {
            shifted
        };
        if (lane as usize) < NUM_WARPS {
            smem[lane as usize] = exclusive;
        }
    }
    block.sync();

    let prefix: T = smem[warp_id as usize];
    Op::combine(prefix, warp_inclusive)
}
