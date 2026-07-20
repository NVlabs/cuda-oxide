/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked-once views for 32-bit kernel indexing.
//!
//! A launch contract proves that thread coordinates fit in `u32`. These views
//! then check one complete element or tile before exposing check-free interior
//! accesses:
//!
//! ```text
//! thread 0 owns [0 .. N)
//! thread 1 owns [N .. 2N)
//! thread 2 owns [2N .. 3N)
//! ```
//!
//! The pointer fields are private. Safe code can obtain a view only through a
//! checked slice constructor or a `DisjointSlice` method that consumes a
//! thread-unique [`crate::thread::ThreadIndex32`].

use crate::DisjointSlice;
use crate::thread::{Index1D, ThreadCoord2D32, ThreadIndex32};
use core::marker::PhantomData;
use core::mem::size_of;

/// Index-space marker where thread `t` owns `N` consecutive elements.
///
/// ```text
/// tile_start = t * N
/// tile_end   = tile_start + N
/// ```
pub enum LinearTiles<const N: usize> {}

/// Index-space marker for a per-thread `ROWS × COLS` row-major tile.
///
/// Thread coordinate `(y, x)` owns rows `y * ROWS..(y + 1) * ROWS` and
/// columns `x * COLS..(x + 1) * COLS`.
///
/// `ROW_STRIDE` is the caller-declared logical pitch: the number of elements
/// from the start of one row to the start of the next. It must match the
/// buffer's layout. It is encoded in the type, so two layouts with different
/// pitches cannot exchange tile proofs. The final row may be partial; each
/// requested tile is checked against the slice length.
pub enum RowMajorTiles<const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> {}

/// A checked local index into a static `N`-element view.
///
/// The private field prevents safe code from inventing an out-of-range value.
/// Use [`new`](Self::new) for a runtime index or [`constant`](Self::constant)
/// for an index that should fold at compile time.
#[must_use]
pub struct LocalIndex32<const N: usize> {
    raw: u32,
}

impl<const N: usize> LocalIndex32<N> {
    /// Check a runtime local index.
    #[inline(always)]
    pub const fn new(raw: u32) -> Option<Self> {
        if N != 0 && N <= u32::MAX as usize && (raw as usize) < N {
            Some(Self { raw })
        } else {
            None
        }
    }

    /// Construct a compile-time local index.
    ///
    /// An invalid constant fails compilation: `N` must be non-zero, `N` must
    /// fit in `u32`, and `I` must be less than `N`. A valid monomorphized call
    /// folds to the immediate index.
    #[inline(always)]
    pub const fn constant<const I: u32>() -> Self {
        const {
            assert!(N != 0, "a static view cannot have zero elements");
            assert!(N <= u32::MAX as usize, "a static view must fit in u32");
            assert!((I as usize) < N, "local index is outside the static view");
        }
        Self { raw: I }
    }

    /// Return the local index.
    #[inline(always)]
    pub const fn get(&self) -> u32 {
        self.raw
    }
}

/// A parent-bound proof that one immutable element is in bounds.
///
/// The proof stores the resolved pointer rather than a free-standing numeric
/// index, so it cannot be applied to an unrelated shorter slice.
#[must_use]
pub struct InBounds32<'a, T> {
    ptr: *const T,
    _borrow: PhantomData<&'a T>,
}

impl<'a, T> InBounds32<'a, T> {
    #[inline(always)]
    unsafe fn from_ptr(ptr: *const T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
        }
    }

    /// Borrow the proven element.
    #[inline(always)]
    pub fn get(&self) -> &T {
        // SAFETY: constructors check the whole parent view before resolving
        // this pointer, and the lifetime is tied to that parent borrow.
        unsafe { &*self.ptr }
    }

    /// Load a `Copy` value from the proven element.
    #[inline(always)]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        *self.get()
    }
}

/// A parent-bound proof that one mutable element is in bounds.
///
/// This capability owns the exclusive borrow of its parent view. It is neither
/// `Copy` nor `Clone`, and its pointer is private.
#[must_use]
pub struct InBoundsMut32<'a, T> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut T>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T> InBoundsMut32<'a, T> {
    #[inline(always)]
    unsafe fn from_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Borrow the proven element for reading.
    #[inline(always)]
    pub fn get(&self) -> &T {
        // SAFETY: the capability carries the exclusive parent borrow.
        unsafe { &*self.ptr }
    }

    /// Borrow the proven element for writing.
    #[inline(always)]
    pub fn get_mut(&mut self) -> &mut T {
        // SAFETY: the capability is unique and carries the exclusive borrow.
        unsafe { &mut *self.ptr }
    }

    /// Load a `Copy` value from the proven element.
    #[inline(always)]
    pub fn read(&self) -> T
    where
        T: Copy,
    {
        *self.get()
    }

    /// Store a value into the proven element.
    #[inline(always)]
    pub fn write(&mut self, value: T) {
        *self.get_mut() = value;
    }
}

/// An immutable `N`-element view checked once at construction.
#[must_use]
pub struct StaticView32<'a, T, const N: usize> {
    ptr: *const T,
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T, const N: usize> StaticView32<'a, T, N> {
    /// Create a view when the slice contains exactly `N` elements and `N` fits
    /// in a 32-bit local index. Zero-length static views are rejected.
    #[inline(always)]
    pub fn from_slice(slice: &'a [T]) -> Option<Self> {
        if N == 0 || N > u32::MAX as usize || slice.len() != N {
            return None;
        }
        Some(Self {
            ptr: slice.as_ptr(),
            _borrow: PhantomData,
        })
    }

    /// Resolve a checked local index with no further bounds branch.
    #[inline(always)]
    pub fn at(&self, index: LocalIndex32<N>) -> InBounds32<'_, T> {
        // SAFETY: LocalIndex32<N> proves index < N, and construction proved
        // that the parent contains exactly N elements.
        unsafe { InBounds32::from_ptr(self.ptr.add(index.get() as usize)) }
    }

    /// Resolve a compile-time local index.
    #[inline(always)]
    pub fn at_const<const I: u32>(&self) -> InBounds32<'_, T> {
        self.at(LocalIndex32::constant::<I>())
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        N as u32
    }

    /// Static views are never empty: `N == 0` is rejected at construction.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

/// A mutable `N`-element view checked once at construction.
///
/// After construction, [`at`](Self::at) and [`at_const`](Self::at_const) use a
/// `LocalIndex32<N>` proof and emit no dynamic bounds check.
#[must_use]
pub struct StaticViewMut32<'a, T, const N: usize> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

/// A checked `ROWS × COLS` mutable tile in a row-major parent allocation.
///
/// The runtime representation is one pointer. Dimensions and row stride live
/// in the type, and construction checks the whole rectangle once.
/// `ROW_STRIDE` is the caller-declared logical pitch and must match the parent
/// buffer's layout:
///
/// ```text
/// base ── row 0: [ COLS elements ] ... stride gap
///         row 1: [ COLS elements ] ... stride gap
///         ...
/// ```
///
/// Interior [`at`](Self::at) calls use only `LocalIndex32` proofs and perform
/// no bounds branch.
#[must_use]
pub struct StaticTileMut32<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize> {
    ptr: *mut T,
    _borrow: PhantomData<&'a mut [T]>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    StaticTileMut32<'a, T, ROWS, COLS, ROW_STRIDE>
{
    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve checked local row/column indices without another bounds branch.
    #[inline(always)]
    pub fn at(&mut self, row: LocalIndex32<ROWS>, col: LocalIndex32<COLS>) -> InBoundsMut32<'_, T> {
        // Tile construction proved that even the largest local offset fits in
        // u32. Wrapping operations state that proof without adding overflow
        // branches; only ptr.add widens the final offset.
        let offset = row
            .get()
            .wrapping_mul(ROW_STRIDE as u32)
            .wrapping_add(col.get());
        // SAFETY: both local indices are in range, and the complete rectangle
        // was checked by tile_2d32 before this wrapper was constructed.
        unsafe { InBoundsMut32::from_ptr(self.ptr.add(offset as usize)) }
    }

    /// Resolve compile-time row/column indices.
    #[inline(always)]
    pub fn at_const<const ROW: u32, const COL: u32>(&mut self) -> InBoundsMut32<'_, T> {
        self.at(
            LocalIndex32::constant::<ROW>(),
            LocalIndex32::constant::<COL>(),
        )
    }

    /// Number of logical rows in the tile.
    #[inline(always)]
    pub const fn rows(&self) -> u32 {
        ROWS as u32
    }

    /// Number of logical columns in the tile.
    #[inline(always)]
    pub const fn cols(&self) -> u32 {
        COLS as u32
    }
}

impl<'a, T, const N: usize> StaticViewMut32<'a, T, N> {
    /// Create a mutable view over exactly `N` elements.
    ///
    /// Zero-length views, widths larger than `u32`, and zero-sized element
    /// types are rejected. Zero-sized mutable tiles would give different GPU
    /// threads the same address, which cannot support exclusive references.
    #[inline(always)]
    pub fn from_slice(slice: &'a mut [T]) -> Option<Self> {
        if !valid_mutable_extent::<T, N>() || slice.len() != N {
            return None;
        }
        Some(Self {
            ptr: slice.as_mut_ptr(),
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        })
    }

    #[inline(always)]
    unsafe fn from_checked_ptr(ptr: *mut T) -> Self {
        Self {
            ptr,
            _borrow: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Resolve a checked local index with no further bounds branch.
    #[inline(always)]
    pub fn at(&mut self, index: LocalIndex32<N>) -> InBoundsMut32<'_, T> {
        // SAFETY: LocalIndex32<N> proves index < N, and construction proved
        // the complete N-element range in bounds.
        unsafe { InBoundsMut32::from_ptr(self.ptr.add(index.get() as usize)) }
    }

    /// Resolve a compile-time local index.
    #[inline(always)]
    pub fn at_const<const I: u32>(&mut self) -> InBoundsMut32<'_, T> {
        self.at(LocalIndex32::constant::<I>())
    }

    /// Number of elements in the view.
    #[inline(always)]
    pub const fn len(&self) -> u32 {
        N as u32
    }

    /// Static views are never empty: `N == 0` is rejected at construction.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

#[inline(always)]
const fn valid_mutable_extent<T, const N: usize>() -> bool {
    N != 0 && N <= u32::MAX as usize && size_of::<T>() != 0
}

#[inline(always)]
const fn valid_row_major_shape<T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>()
-> bool {
    ROWS != 0
        && COLS != 0
        && ROW_STRIDE != 0
        && ROWS <= u32::MAX as usize
        && COLS <= u32::MAX as usize
        && ROW_STRIDE <= u32::MAX as usize
        && ROW_STRIDE >= COLS
        && size_of::<T>() != 0
}

const INVALID_LINEAR_2D: u64 = u64::MAX;

/// Compute `row * stride + col` without an `Option` aggregate. Returning the
/// sentinel keeps the device representation scalar; callers reject it before
/// converting back to a pointer offset.
#[inline(always)]
fn checked_linear_2d(row: u32, stride: u32, col: u32) -> u64 {
    if stride == 0 || row > (u32::MAX - col) / stride {
        INVALID_LINEAR_2D
    } else {
        u64::from(row * stride + col)
    }
}

#[inline(always)]
fn scaled_tile_axis_fits(origin: u32, width: u32) -> bool {
    width != 0 && origin <= (u32::MAX - (width - 1)) / width
}

impl<'a, T> DisjointSlice<'a, T, Index1D> {
    /// Check this thread's 32-bit element index once and return a parent-bound
    /// read/write capability.
    #[inline(always)]
    pub fn element_thread32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> Option<InBoundsMut32<'_, T>> {
        if size_of::<T>() == 0 {
            return None;
        }
        let index = thread.get() as usize;
        if index >= self.len() {
            return None;
        }
        // SAFETY: the single bounds check above covers the resolved element.
        // ThreadIndex32 is unique for the validated 1D launch and is consumed.
        Some(unsafe { InBoundsMut32::from_ptr(self.as_mut_ptr().add(index)) })
    }
}

impl<'a, T, const N: usize> DisjointSlice<'a, T, LinearTiles<N>> {
    /// Check one complete per-thread tile and return a check-free static view.
    ///
    /// ```text
    /// prove thread * N + (N - 1) <= u32::MAX
    /// start = thread * N
    /// last  = start + (N - 1)
    /// accept only when last < slice.len()
    /// ```
    #[inline(always)]
    pub fn tile_thread32<'kernel>(
        &mut self,
        thread: ThreadIndex32<'kernel>,
    ) -> Option<StaticViewMut32<'_, T, N>> {
        if size_of::<T>() == 0 {
            return None;
        }
        let start = checked_linear_tile_start::<N>(thread.get(), self.len());
        if start == u64::MAX {
            return None;
        }
        let start = start as u32;
        // SAFETY: start..end was computed without overflow and checked as one
        // complete range. Consuming the unique thread index makes tiles
        // disjoint, and the non-ZST check gives each element an address.
        Some(unsafe { StaticViewMut32::from_checked_ptr(self.as_mut_ptr().add(start as usize)) })
    }
}

impl<'a, T, const ROWS: usize, const COLS: usize, const ROW_STRIDE: usize>
    DisjointSlice<'a, T, RowMajorTiles<ROWS, COLS, ROW_STRIDE>>
{
    /// Check one complete rectangular tile and return a check-free static view.
    ///
    /// `ROW_STRIDE` is the caller-declared logical row pitch and must match the
    /// buffer's layout. The slice length does not have to be a multiple of the
    /// pitch; a tile is returned only when its complete rectangle fits.
    ///
    /// Construction proves the following before creating a pointer:
    ///
    /// ```text
    /// start_row = thread.row * ROWS
    /// start_col = thread.col * COLS
    /// last_col  < ROW_STRIDE       (the tile cannot wrap into the next row)
    /// last_row * ROW_STRIDE + last_col < parent.len()
    /// ```
    #[inline(always)]
    pub fn tile_2d32<'kernel>(
        &mut self,
        thread: ThreadCoord2D32<'kernel>,
    ) -> Option<StaticTileMut32<'_, T, ROWS, COLS, ROW_STRIDE>> {
        let start = checked_row_major_tile_start::<T, ROWS, COLS, ROW_STRIDE>(
            thread.row(),
            thread.col(),
            self.len(),
        );
        if start == INVALID_LINEAR_2D {
            return None;
        }

        // SAFETY: the scalar-only helper checked the complete rectangle.
        // Distinct 2D thread coordinates map to disjoint row and column bands.
        Some(unsafe { StaticTileMut32::from_checked_ptr(self.as_mut_ptr().add(start as usize)) })
    }
}

/// Check a complete row-major tile without accepting or returning a pointer.
///
/// Keeping this proof scalar-only lets `tile_2d32` MIR-inline as a tiny pointer
/// wrapper. The kernel's original global pointer provenance therefore reaches
/// the final loads and stores before LLVM capture/address-space inference.
#[inline(always)]
fn checked_row_major_tile_start<
    T,
    const ROWS: usize,
    const COLS: usize,
    const ROW_STRIDE: usize,
>(
    thread_row: u32,
    thread_col: u32,
    len: usize,
) -> u64 {
    if !valid_row_major_shape::<T, ROWS, COLS, ROW_STRIDE>() {
        return INVALID_LINEAR_2D;
    }

    let rows = ROWS as u32;
    let cols = COLS as u32;
    let stride = ROW_STRIDE as u32;
    if !scaled_tile_axis_fits(thread_row, rows) || !scaled_tile_axis_fits(thread_col, cols) {
        return INVALID_LINEAR_2D;
    }

    let start_row = thread_row * rows;
    let start_col = thread_col * cols;
    let last_row = start_row + (rows - 1);
    let last_col = start_col + (cols - 1);

    // The X tile must remain inside one logical row. This also makes tiles
    // owned by adjacent X threads disjoint.
    if last_col >= stride {
        return INVALID_LINEAR_2D;
    }

    let start = checked_linear_2d(start_row, stride, start_col);
    let last = checked_linear_2d(last_row, stride, last_col);
    if start == INVALID_LINEAR_2D || last == INVALID_LINEAR_2D || (last as usize) >= len {
        INVALID_LINEAR_2D
    } else {
        start
    }
}

/// Keep the range proof scalar-only so the pointer-bearing wrapper above stays
/// small enough to inline without losing the kernel parameter's global-memory
/// provenance.
#[inline(always)]
fn checked_linear_tile_start<const N: usize>(thread: u32, len: usize) -> u64 {
    if N == 0 || N > u32::MAX as usize {
        return u64::MAX;
    }
    let width = N as u32;
    let last_offset = width - 1;
    // Prove both arithmetic operations before performing either one. We check
    // the inclusive last element so a one-element tile at u32::MAX remains
    // representable.
    if thread > (u32::MAX - last_offset) / width {
        return u64::MAX;
    }
    let start = thread * width;
    let last = start + last_offset;
    if (last as usize) < len {
        u64::from(start)
    } else {
        u64::MAX
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_indices_reject_invalid_extents_and_offsets() {
        assert!(LocalIndex32::<0>::new(0).is_none());
        assert!(LocalIndex32::<4>::new(3).is_some());
        assert!(LocalIndex32::<4>::new(4).is_none());
    }

    #[test]
    fn immutable_static_view_reads_after_one_shape_check() {
        let values = [10_u32, 20, 30, 40];
        let view = StaticView32::<_, 4>::from_slice(&values).unwrap();
        assert_eq!(view.at_const::<2>().read(), 30);
    }

    #[test]
    fn mutable_static_view_writes_after_one_shape_check() {
        let mut values = [0_u32; 4];
        let mut view = StaticViewMut32::<_, 4>::from_slice(&mut values).unwrap();
        view.at_const::<3>().write(9);
        assert_eq!(values[3], 9);
    }

    #[test]
    fn mutable_static_view_rejects_zero_sized_elements() {
        let mut values = [(); 4];
        assert!(StaticViewMut32::<_, 4>::from_slice(&mut values).is_none());
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn tile_range_keeps_the_exact_u32_boundary_and_rejects_overflow() {
        let full_u32_len = u32::MAX as usize + 1;
        assert_eq!(
            checked_linear_tile_start::<1>(u32::MAX, full_u32_len),
            u64::from(u32::MAX)
        );
        assert_eq!(
            checked_linear_tile_start::<1>(u32::MAX, u32::MAX as usize),
            u64::MAX
        );
        assert_eq!(
            checked_linear_tile_start::<2>(u32::MAX, full_u32_len),
            u64::MAX
        );
    }

    #[test]
    fn row_major_shape_rejects_empty_wide_and_zero_sized_tiles() {
        assert!(!valid_row_major_shape::<u32, 0, 4, 8>());
        assert!(!valid_row_major_shape::<u32, 2, 9, 8>());
        assert!(!valid_row_major_shape::<(), 2, 4, 8>());
        assert!(valid_row_major_shape::<u32, 2, 4, 8>());
        #[cfg(target_pointer_width = "64")]
        assert!(!valid_row_major_shape::<u32, { u32::MAX as usize + 1 }, 1, 1>());
    }

    #[test]
    fn scalar_linear_helper_uses_a_reserved_failure_value() {
        assert_eq!(checked_linear_2d(2, 10, 3), 23);
        assert_eq!(checked_linear_2d(1, 0, 0), INVALID_LINEAR_2D);
        assert_eq!(checked_linear_2d(u32::MAX, 1, 0), u32::MAX as u64);
        assert_eq!(
            checked_linear_2d(u32::MAX, u32::MAX, u32::MAX),
            INVALID_LINEAR_2D
        );
    }

    #[test]
    fn row_major_tile_helper_checks_the_complete_rectangle() {
        assert_eq!(checked_row_major_tile_start::<u32, 2, 4, 16>(1, 1, 56), 36);
        assert_eq!(
            checked_row_major_tile_start::<u32, 2, 4, 16>(1, 1, 55),
            INVALID_LINEAR_2D
        );
        assert_eq!(checked_row_major_tile_start::<u32, 1, 2, 64>(0, 31, 64), 62);
        assert_eq!(
            checked_row_major_tile_start::<u32, 1, 2, 64>(0, 32, 128),
            INVALID_LINEAR_2D
        );
        assert_eq!(
            checked_row_major_tile_start::<u32, 2, 1, 1>(u32::MAX, 0, usize::MAX),
            INVALID_LINEAR_2D
        );
        assert_eq!(
            checked_row_major_tile_start::<(), 1, 1, 1>(0, 0, 1),
            INVALID_LINEAR_2D
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn row_major_tile_helper_keeps_the_exact_u32_linear_boundary() {
        assert_eq!(
            checked_row_major_tile_start::<u32, 1, 1, 1>(u32::MAX, 0, u32::MAX as usize + 1,),
            u32::MAX as u64
        );
    }

    #[test]
    fn tile_axis_precheck_keeps_the_exact_u32_boundary() {
        assert!(scaled_tile_axis_fits(u32::MAX, 1));
        assert!(scaled_tile_axis_fits(u32::MAX / 2, 2));
        assert!(!scaled_tile_axis_fits(u32::MAX / 2 + 1, 2));
        assert!(!scaled_tile_axis_fits(0, 0));
    }

    #[test]
    fn static_tile_wrapper_is_one_pointer() {
        assert_eq!(
            size_of::<StaticTileMut32<'_, u32, 2, 4, 16>>(),
            size_of::<*mut u32>()
        );
    }
}
