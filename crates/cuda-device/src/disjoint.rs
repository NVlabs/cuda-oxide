/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! DisjointSlice - a type-safe abstraction for parallel GPU writes.
//!
//! This module provides `DisjointSlice<T>`, which guarantees that each thread
//! accesses a unique element, preventing data races.
//!
//! # Safety Model
//!
//! Safety is enforced through the type system and bounds checking:
//!
//! 1. **ThreadIndex**: Can only be constructed by `index_1d`,
//!    `index_2d::<S>`, or the unsafe `index_2d_runtime`, which derive
//!    the index from hardware built-in variables (`threadIdx`,
//!    `blockIdx`, `blockDim`) -- read-only special registers assigned
//!    by the runtime at kernel launch. The formula combines these into
//!    a scalar index per thread. 2D stride is encoded in the index
//!    space, so mixing const strides is rejected at compile time.
//!
//! 2. **`get_mut(idx)`**: Bounds-checked access via an explicit
//!    `ThreadIndex`. Returns `Option<&mut T>` — `None` for out-of-bounds
//!    threads.
//!
//! 3. **`get_mut_indexed()`**: One-call form that mints the witness and
//!    resolves it in a single shot. Available when the index space implements
//!    [`crate::thread::IndexFormula`] (i.e. `Index1D` or `Index2D<S>`).
//!
//! 4. **`get_unchecked_mut()`**: Unsafe escape hatch for performance-critical
//!    paths where bounds have been validated by other means.
//!
//! The unsafe boundary is pushed away from each access. A prepared launch proves
//! that the kernel's declared index domain matches its geometry; an unprepared
//! raw launch is unsafe and leaves that proof to the caller. Constructing a
//! `DisjointSlice` from raw memory is also unsafe.

use crate::thread::{DisjointBlock, Index1D, IndexFormula, LaunchContext, ThreadIndex};
use crate::view::{LinearTiles, RowMajorTiles, RuntimeRowMajorTiles};
use core::marker::PhantomData;
use core::mem::size_of;

/// A slice-like type that can only be accessed with thread-local indices.
///
/// # Safety Invariants
///
/// The type system enforces these invariants:
/// 1. Default access via `get_mut(ThreadIndex)` is bounds-checked and sound.
/// 2. `ThreadIndex` can only be created by trusted index functions
///    (`index_1d`, `index_2d::<S>`, `unsafe index_2d_runtime`), which
///    derive the index from hardware built-in variables -- read-only
///    special registers assigned by the runtime at launch.
/// 3. Each thread's `ThreadIndex` is unique within its index space when the
///    prepared launch contract matches that space. Raw launch callers must
///    uphold the same condition explicitly.
///
/// Each thread accesses a unique element, making parallel writes safe without
/// synchronization.
///
/// # Memory Layout
///
/// Internally, this is identical to a slice: `{ ptr: *mut T, len: usize }`
/// The safety comes from type-level enforcement and bounds checking.
///
/// # Soundness
///
/// `get_mut()` returns `Option<&mut T>`, making out-of-bounds access
/// impossible in safe code. The previous API (`get() -> &mut T`) relied on
/// the caller to check bounds externally; in release builds this was UB for
/// out-of-bounds indices — a soundness hole. The current design follows
/// `slice::get_mut()` / `slice::get_unchecked_mut()` from std: the safe
/// path is sound when reached through a prepared launch. Raw launch geometry
/// and the unchecked element accessor are explicit unsafe escape hatches.
///
/// The type is `Send` but NOT `Sync`: each GPU thread gets its own copy of
/// the struct (with the same backing pointer), then uses its unique
/// `ThreadIndex` to access a different element. Sharing `&DisjointSlice`
/// across threads is not meaningful.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::{thread, DisjointSlice};
///
/// #[kernel]
/// pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = thread::index_1d();
///     let i = idx.get();
///     if let Some(c_elem) = c.get_mut(idx) {
///         *c_elem = a[i] + b[i];
///     }
/// }
/// ```
#[repr(C)]
pub struct DisjointSlice<'a, T, IndexSpace = Index1D> {
    ptr: *mut T,
    len: usize,
    _marker: PhantomData<&'a mut [T]>,
    _space: PhantomData<fn() -> IndexSpace>,
}

mod launch_contract_sealed {
    pub trait Sealed {}
}

impl<'a, T, IndexSpace> launch_contract_sealed::Sealed for DisjointSlice<'a, T, IndexSpace> {}

/// Compiler-facing proof that a `DisjointSlice` has the expected element type
/// and supports a launch domain.
///
/// This trait is sealed: only the genuine [`DisjointSlice`] type can implement
/// it. `#[cuda_module]` adds this bound to contracted kernels so Rust resolves
/// type aliases before checking the declared launch domain.
///
/// A 2D index space also supports a 1D launch (all Y dimensions are one), but
/// a 1D index space cannot support a 2D launch.
#[doc(hidden)]
pub trait __LaunchContractDisjointSlice<Element, const DOMAIN: u8>:
    launch_contract_sealed::Sealed
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 1> for DisjointSlice<'a, T, Index1D> {}

impl<'a, T, const N: usize> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, LinearTiles<N>>
{
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, RowMajorTiles<ROWS, COLS, ROW_STRIDE>>
{
}

impl<'a, T, const ROWS: usize, const COLS: usize> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, RuntimeRowMajorTiles<ROWS, COLS>>
{
}

impl<'a, T, const ROW_STRIDE: usize> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, crate::thread::Index2D<ROW_STRIDE>>
{
}

impl<'a, T, const ROW_STRIDE: usize> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, crate::thread::Index2D<ROW_STRIDE>>
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 1>
    for DisjointSlice<'a, T, crate::thread::Runtime2DIndex>
{
}

impl<'a, T> __LaunchContractDisjointSlice<T, 2>
    for DisjointSlice<'a, T, crate::thread::Runtime2DIndex>
{
}

impl<'a, T, IndexSpace> DisjointSlice<'a, T, IndexSpace> {
    /// Create a DisjointSlice from a raw pointer and length.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` points to valid, aligned memory for `len` elements of type `T`
    /// - The memory will remain valid and not be deallocated for lifetime `'a`
    /// - **Exclusive access**: no other live `DisjointSlice<T>` (or `&mut [T]`,
    ///   `&[T]`, raw read/write) covers any byte of
    ///   `ptr..ptr + len * size_of::<T>()` for the duration of `'a`. Two
    ///   `DisjointSlice` over the same range gives every thread two `&mut T`
    ///   to the same slot, which is UB regardless of the witness-type story.
    /// - The kernel launch configuration ensures threads access disjoint elements
    ///   (i.e., grid dimensions match the data dimensions)
    #[inline]
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        DisjointSlice {
            ptr,
            len,
            _marker: PhantomData,
            _space: PhantomData,
        }
    }

    /// Create a DisjointSlice from a mutable slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure that:
    /// - The kernel launch configuration ensures threads access disjoint elements
    /// - No other code accesses the slice during kernel execution
    #[inline]
    pub unsafe fn from_mut_slice(slice: &'a mut [T]) -> Self {
        unsafe { Self::from_raw_parts(slice.as_mut_ptr(), slice.len()) }
    }

    /// Get the length of the slice.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the slice is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get a mutable reference to an element at a thread-local index,
    /// returning `None` if the index is out of bounds.
    ///
    /// This is the default, sound access method. Mirrors `slice::get_mut()`.
    ///
    /// # Safety Argument
    ///
    /// This method is safe (not marked `unsafe`) because:
    ///
    /// 1. **Bounds checked**: Returns `None` for out-of-bounds indices.
    ///
    /// 2. **Unique access**: `ThreadIndex` can only be constructed by
    ///    `index_1d()`, `index_2d::<S>()`, or the unsafe
    ///    `index_2d_runtime()`, which derive the index from hardware
    ///    built-in variables (`threadIdx`, `blockIdx`, `blockDim`) --
    ///    read-only special registers assigned by the runtime at kernel
    ///    launch. 2D stride is carried in the index space, so a slice
    ///    can only be indexed by a matching witness.
    ///
    /// 3. **No data races**: Given the constraint above, each thread's
    ///    `ThreadIndex` is unique under the prepared launch's matching index
    ///    domain. An unsafe raw launch takes responsibility for the same
    ///    geometry invariant, so each thread accesses a different location.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let idx = thread::index_1d();
    /// let i = idx.get();
    /// if let Some(elem) = c.get_mut(idx) {
    ///     *elem = a[i] + b[i];
    /// }
    /// ```
    /// Bounds-checked access to the `N` consecutive elements a thread owns.
    ///
    /// Takes the tile proof from [`ThreadIndex::scale`] and returns that thread's
    /// elements as a fixed-size array, or `None` if the tile does not fit.
    ///
    /// This is the safe form of the multi-element write that otherwise needs
    /// `get_unchecked_mut` plus a hand-discharged bounds argument. Writing four
    /// elements per thread becomes:
    ///
    /// ```rust,ignore
    /// let block = thread::index_1d().scale::<4>();
    /// if let Some(row) = output.get_block_mut(block) {
    ///     *row = [o0, o1, o2, o3];
    /// }
    /// ```
    ///
    /// Returning `&mut [T; N]` rather than `&mut [T]` is deliberate: the length
    /// is static, so the tile is a value the caller can assign in one statement.
    ///
    /// It does **not** widen the access, and it is worth being explicit about
    /// that. Measured on sm_86, whether an access becomes `LDG.E.128` /
    /// `STG.E.128` is decided by the *alignment of the element type*, not by the
    /// static length: a 16-byte-aligned aggregate fuses, the same payload at
    /// 4-byte alignment does not. Since `T` here is the element type of a
    /// `DisjointSlice<T>`, a `DisjointSlice<f32>` yields `&mut [f32; N]` with
    /// 4-byte alignment, which stays scalar.
    ///
    /// To get wide accesses, make the element type aligned and use
    /// [`Self::get_mut`] on a `DisjointSlice<F32x4>` where
    /// `#[repr(C, align(16))] struct F32x4([f32; 4])`. That path emits one
    /// `LDG.E.128` plus one `STG.E.128` with no `unsafe` and no inline PTX; see
    /// the `vectorization` example. This method is about the disjointness proof
    /// for multi-element access, not about access width.
    ///
    /// # Soundness
    ///
    /// Two threads cannot obtain overlapping tiles. Distinct `ThreadIndex`
    /// values scale to distinct `N`-sized tiles that partition the space, and
    /// [`DisjointBlock`] is neither `Copy` nor `Clone`, so one thread cannot
    /// hold two proofs covering the same elements.
    #[inline]
    pub fn get_block_mut<'kernel, const N: usize>(
        &mut self,
        block: DisjointBlock<'kernel, N, IndexSpace>,
    ) -> Option<&mut [T; N]> {
        if size_of::<T>() == 0 || !block.is_valid() {
            return None;
        }
        let start = block.start();
        // Checked so a tile straddling the end of the address space cannot wrap
        // into a passing bounds test.
        if start.checked_add(N)? > self.len {
            return None;
        }
        // SAFETY:
        // - The bounds check above proves `start .. start + N` is in range.
        // - `block` came from scaling a `ThreadIndex`, so this thread's tile is
        //   disjoint from every other thread's, and the proof is not
        //   duplicable.
        // - `[T; N]` has the same layout as `N` consecutive `T`, so the cast is
        //   valid for a region this long.
        Some(unsafe { &mut *self.ptr.add(start).cast::<[T; N]>() })
    }

    /// Visit each of the `K` tiles a thread owns, exactly once.
    ///
    /// Takes the grid-stride proof from [`DisjointBlock::repeat`] and hands the
    /// closure one `&mut [T; N]` per pass, with the pass number. Returns how many
    /// tiles were visited, which is `K` unless the buffer ran out first.
    ///
    /// ```rust,ignore
    /// let tiling = thread::index_1d().scale::<4>().repeat::<4>();
    /// let done = output.for_each_tile(tiling, |k, tile| {
    ///     *tile = compute_stripe(k);
    /// });
    /// ```
    ///
    /// # Why a callback rather than an accessor
    ///
    /// A `get_tile_mut(&mut self, k)` taking `k` at runtime could be called
    /// twice with the same `k`, producing two live `&mut` to the same elements -
    /// the disjointness proof covers *different* threads and *different* passes,
    /// not repeated access to one pass. Driving the loop here is what keeps each
    /// tile handed out once, so the guarantee is structural rather than a rule
    /// the caller has to follow.
    ///
    /// # Soundness
    ///
    /// Tiles are pairwise disjoint across both threads and passes, by the
    /// mixed-radix injectivity argument on [`DisjointBlock::repeat`]. Each is
    /// bounds-checked before the closure sees it, and iteration stops at the
    /// first tile that does not fit, so a partially covering launch truncates
    /// instead of writing past the end.
    #[inline]
    pub fn for_each_tile<'kernel, const N: usize, const K: usize>(
        &mut self,
        tiling: crate::thread::DisjointTiling<'kernel, N, K>,
        mut f: impl FnMut(usize, &mut [T; N]),
    ) -> usize {
        if size_of::<T>() == 0 || !tiling.is_valid() {
            return 0;
        }
        let (base, period) = (tiling.start(), tiling.period());
        let mut visited = 0;
        for k in 0..K {
            // Checked throughout: a pass whose offset or end wraps must stop the
            // loop, not fold into a passing bounds test.
            let Some(offset) = period.checked_mul(k) else {
                break;
            };
            let Some(start) = base.checked_add(offset) else {
                break;
            };
            let Some(end) = start.checked_add(N) else {
                break;
            };
            if end > self.len {
                break;
            }
            // SAFETY:
            // - The bounds check above proves `start .. start + N` is in range.
            // - Distinct threads decode to distinct `t` digits and distinct
            //   passes to distinct `k` digits, so this tile overlaps no other
            //   tile handed out anywhere in the launch.
            // - `k` is visited once per loop, so the closure cannot receive two
            //   references to this tile.
            // - `[T; N]` has the layout of `N` consecutive `T`.
            f(k, unsafe { &mut *self.ptr.add(start).cast::<[T; N]>() });
            visited += 1;
        }
        visited
    }

    #[inline]
    pub fn get_mut<'kernel>(&mut self, idx: ThreadIndex<'kernel, IndexSpace>) -> Option<&mut T> {
        let i = idx.get();
        if size_of::<T>() != 0 && idx.is_valid() && i < self.len {
            // SAFETY:
            // - Bounds check passed above.
            // - `idx` is a ThreadIndex derived from hardware built-in variables.
            //   The prepared launch contract, or an unsafe raw launch caller,
            //   guarantees that its index space is unique for this geometry.
            // - The DisjointSlice was constructed with valid memory (from_raw_parts safety).
            Some(unsafe { &mut *self.ptr.add(i) })
        } else {
            None
        }
    }

    /// Get a raw pointer to the underlying data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get a mutable reference to an element at a raw index, without
    /// bounds checking.
    ///
    /// This is an escape hatch for performance-critical paths where bounds
    /// have been validated by other means, such as:
    /// - Warp reductions where only lane 0 writes to a unique warp index
    /// - Histogram updates with atomic operations
    /// - Scatter operations with known-unique destinations
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `idx < self.len()` (bounds are valid)
    /// - No two threads write to the same index simultaneously
    /// - The uniqueness guarantee comes from the algorithm (document it!)
    ///
    /// # Example: Warp Reduction
    ///
    /// ```rust,ignore
    /// // SAFETY: Only lane 0 of each warp writes, and warp indices are unique
    /// if warp::lane_id() == 0 {
    ///     let warp_idx = gid.get() / 32;
    ///     unsafe { *out.get_unchecked_mut(warp_idx) = sum; }
    /// }
    /// ```
    #[inline]
    pub unsafe fn get_unchecked_mut(&mut self, idx: usize) -> &mut T {
        debug_assert!(
            idx < self.len,
            "Index out of bounds: {} >= {}",
            idx,
            self.len
        );
        unsafe { &mut *self.ptr.add(idx) }
    }
}

impl<'a, T, IS: IndexFormula> DisjointSlice<'a, T, IS> {
    /// One-shot indexed access — mints this thread's witness and resolves
    /// it to a mutable reference in a single call.
    ///
    /// Equivalent to:
    ///
    /// ```ignore
    /// let idx = thread::index_*();      // matching the slice's index space
    /// let cell = slice.get_mut(idx);    // None if out of bounds
    /// // (cell, idx)                    // ThreadIndex still in hand
    /// ```
    ///
    /// but with one index computation instead of two, and a flatter
    /// match: out-of-grid threads (e.g. `col >= ROW_STRIDE` for 2D) and
    /// out-of-slice indices both fold into a single `None`.
    ///
    /// # Where you call it
    ///
    /// Inside `#[kernel]` / `#[device]` the macro splices in the kernel
    /// scope for you, so the call site reads:
    ///
    /// ```rust,ignore
    /// #[kernel]
    /// fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    ///     if let Some((c_elem, idx)) = c.get_mut_indexed() {
    ///         let i = idx.get();
    ///         *c_elem = a[i] + b[i];
    ///     }
    /// }
    /// ```
    ///
    /// `get_mut_indexed` is a reserved method name inside annotated
    /// bodies — see the [reserved names note on
    /// `ThreadIndex`](crate::thread::ThreadIndex#reserved-names-inside-kernel-and-device).
    ///
    /// # Index space coverage
    ///
    /// Available for slices whose index space implements [`IndexFormula`]:
    /// `Index1D` and `Index2D<ROW_STRIDE>`. For `Runtime2DIndex` the row
    /// stride is opaque to the type system, so use the unsafe
    /// [`index_2d_runtime`](crate::thread::index_2d_runtime) and
    /// [`get_mut`](Self::get_mut) pair explicitly.
    ///
    /// # Safety Argument
    ///
    /// Same as [`get_mut`](Self::get_mut): the returned `ThreadIndex` is
    /// minted from hardware built-ins via the trusted `__internal::*`
    /// path, the bounds check is explicit, and the borrow of `&mut self`
    /// keeps the returned reference exclusive for its lifetime.
    #[inline]
    pub fn get_mut_indexed<'kernel, Domain, Coordinates>(
        &mut self,
        scope: &'kernel LaunchContext<'kernel, Domain, Coordinates>,
    ) -> Option<(&mut T, ThreadIndex<'kernel, IS>)>
    where
        Domain: crate::thread::__internal::LaunchDomain,
    {
        let idx = IS::from_scope(scope)?;
        let i = idx.get();
        if size_of::<T>() != 0 && i < self.len {
            // SAFETY:
            // - bounds check passed above
            // - idx is freshly minted from hardware special registers (no
            //   laundering — !Copy, !Send, `'kernel`-bound); the prepared
            //   launch, or unsafe raw caller, proves IS is unique here
            // - DisjointSlice was constructed with valid memory
            //   (from_raw_parts safety)
            Some((unsafe { &mut *self.ptr.add(i) }, idx))
        } else {
            None
        }
    }
}

// SAFETY: DisjointSlice can be sent between threads because:
// - A prepared launch, or unsafe raw caller, guarantees that each thread's
//   ThreadIndex selects a unique element for the active geometry
// - The pointer and length are just data, no thread affinity
// - T: Send means the elements themselves can be sent between threads
unsafe impl<'a, T: Send, IndexSpace> Send for DisjointSlice<'a, T, IndexSpace> {}

// DisjointSlice auto-trait summary:
//   Send: yes (explicit impl above, when T: Send)
//   Sync: NO (not implemented) — each GPU thread gets its own copy of the
//         struct, then uses its conditionally unique ThreadIndex to access a
//         different element. Sharing &DisjointSlice across threads would allow
//         multiple threads to call get_mut() on the same struct, which
//         would produce aliasing &mut T references — unsound.

#[cfg(test)]
mod block_tests {
    /// `DisjointBlock` can only come from a `ThreadIndex`, which needs a launch
    /// context, so these tests exercise the arithmetic and bounds logic through
    /// a locally constructed block instead of a real launch.
    ///
    /// `scale` is a pure function of `raw` and `N`, so the tiling property below
    /// is the same one the real witness relies on.
    fn tile(raw: usize, n: usize) -> Option<(usize, usize)> {
        // Mirrors `ThreadIndex::scale`: invalid on N == 0 or overflow.
        if n == 0 {
            return None;
        }
        let start = raw.checked_mul(n)?;
        if start == usize::MAX {
            return None;
        }
        Some((start, start + n))
    }

    /// The soundness claim: distinct thread indices scale to non-overlapping
    /// ranges. This is what lets `get_block_mut` hand out `&mut [T; N]` to every
    /// thread at once.
    #[test]
    fn distinct_indices_produce_non_overlapping_tiles() {
        for n in [1usize, 2, 4, 8, 16] {
            let mut ranges = [(0usize, 0usize); 64];
            for (slot, raw) in (0..64).enumerate() {
                ranges[slot] = tile(raw, n).expect("small tiles must be valid");
            }
            for i in 0..ranges.len() {
                for j in (i + 1)..ranges.len() {
                    let (a_lo, a_hi) = ranges[i];
                    let (b_lo, b_hi) = ranges[j];
                    assert!(
                        a_hi <= b_lo || b_hi <= a_lo,
                        "tiles for raw={i} and raw={j} overlap at N={n}: \
                         [{a_lo},{a_hi}) vs [{b_lo},{b_hi})"
                    );
                }
            }
        }
    }

    /// Tiles must also be gapless, otherwise scaling would silently skip
    /// elements rather than partition the range.
    #[test]
    fn tiles_are_contiguous_and_gapless() {
        let n = 4;
        let mut expected_start = 0;
        for raw in 0..32 {
            let (start, end) = tile(raw, n).unwrap();
            assert_eq!(start, expected_start, "gap before raw={raw}");
            assert_eq!(end - start, n);
            expected_start = end;
        }
    }

    /// A multiplication that overflows must invalidate the witness rather than
    /// wrap into a small, passing start offset.
    #[test]
    fn overflowing_scale_is_rejected() {
        assert_eq!(tile(usize::MAX / 2, 4), None);
        assert_eq!(tile(usize::MAX, 2), None);
        // N == 0 would map every thread to start 0.
        assert_eq!(tile(7, 0), None);
    }

    /// `start + N` must be checked, so a tile straddling the end of the address
    /// space cannot wrap into a passing bounds test.
    #[test]
    fn bounds_test_uses_checked_addition() {
        let len = 64usize;
        // Well inside.
        assert!(tile(0, 4).unwrap().1 <= len);
        assert!(tile(15, 4).unwrap().1 <= len);
        // Exactly at the end is still in range.
        assert_eq!(tile(15, 4).unwrap().1, len);
        // One tile past the end must not fit.
        assert!(tile(16, 4).unwrap().1 > len);
        // And the wrapping case is rejected outright.
        assert_eq!(usize::MAX.checked_add(4), None);
    }
}

#[cfg(test)]
mod tiling_algebra_tests {
    //! Tests for the mixed-radix argument behind `DisjointBlock::repeat`.
    //!
    //! `repeat` reads the grid extent from hardware registers, so these exercise
    //! the arithmetic directly: `phi` below is the same index expression, and the
    //! properties asserted are the ones the safety of `for_each_tile` rests on.

    /// The index expression: `k·(G·N) + t·N + c`.
    fn phi(k: usize, t: usize, c: usize, grid: usize, n: usize) -> usize {
        k * (grid * n) + t * n + c
    }

    /// The claimed inverse: recover each digit by division and remainder.
    fn decode(index: usize, grid: usize, n: usize) -> (usize, usize, usize) {
        (index / (grid * n), (index / n) % grid, index % n)
    }

    /// `phi` is injective on the digit domain, which is the whole proof: an index
    /// determines the thread it came from, so two threads cannot share one.
    #[test]
    fn index_expression_is_injective() {
        for &(grid, n, k_max) in &[(4usize, 1usize, 3usize), (4, 4, 3), (8, 2, 4), (3, 5, 2)] {
            for k in 0..k_max {
                for t in 0..grid {
                    for c in 0..n {
                        let v = phi(k, t, c, grid, n);
                        assert_eq!(
                            decode(v, grid, n),
                            (k, t, c),
                            "decode failed for k={k} t={t} c={c} (G={grid}, N={n})"
                        );
                    }
                }
            }
        }
    }

    /// The consequence used by `for_each_tile`: tiles belonging to different
    /// threads never overlap, across every pass.
    #[test]
    fn tiles_of_distinct_threads_never_overlap() {
        let (grid, n, passes) = (8usize, 4usize, 3usize);
        let mut owner = [usize::MAX; 8 * 4 * 3];
        for k in 0..passes {
            for t in 0..grid {
                for c in 0..n {
                    let v = phi(k, t, c, grid, n);
                    assert_eq!(
                        owner[v],
                        usize::MAX,
                        "element {v} claimed twice: by thread {} and thread {t}",
                        owner[v]
                    );
                    owner[v] = t;
                }
            }
        }
        // Every element in the covered span has exactly one owner.
        assert!(
            owner.iter().all(|&o| o != usize::MAX),
            "grid-stride passes must tile the span with no gaps"
        );
    }

    /// The reason the period is derived rather than supplied. With a period
    /// smaller than `G·N` the thread and pass digits overlap and two threads
    /// collide on the same element - so a `repeat_with_stride` taking a
    /// caller-chosen value would be unsound.
    #[test]
    fn a_period_below_grid_span_collides() {
        let (grid, n) = (4usize, 1usize);
        let bad_period = 2; // G·N would be 4.
        // (k=0, t=2) and (k=1, t=0) both land on element 2.
        let a = 0 * bad_period + 2 * n;
        let b = 1 * bad_period + 0 * n;
        assert_eq!(a, b, "the counterexample must actually collide");

        // The derived period keeps them apart.
        let good_period = grid * n;
        assert_ne!(
            0 * good_period + 2 * n,
            1 * good_period + 0 * n,
            "period G·N must separate the pass and thread digits"
        );
    }

    /// A period larger than `G·N` stays injective but leaves gaps, so it is
    /// sound yet wasteful. Recorded to show the derived period is the tight
    /// choice, not merely a safe one.
    #[test]
    fn a_period_above_grid_span_is_injective_but_leaves_gaps() {
        let (grid, n, passes) = (4usize, 1usize, 2usize);
        let period = grid * n + 1;
        let mut seen = [false; 16];
        for k in 0..passes {
            for t in 0..grid {
                let v = k * period + t * n;
                assert!(!seen[v], "still injective");
                seen[v] = true;
            }
        }
        let covered = seen.iter().filter(|&&s| s).count();
        assert_eq!(covered, grid * passes);
        // Element 4 falls in the gap: no (k, t) reaches it.
        assert!(!seen[4], "an oversized period must leave the span gapped");
    }
}
