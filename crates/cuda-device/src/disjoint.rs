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
//! The unsafe boundary is pushed to the construction of `DisjointSlice` from raw
//! memory, not to the per-access level.

use crate::thread::{Index1D, IndexFormula, KernelScope, ThreadIndex};
use core::marker::PhantomData;

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
/// 3. Each thread's `ThreadIndex` is unique within its index space.
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
/// path is always sound, and the unsafe escape hatch (`get_unchecked_mut`)
/// is explicitly opted into.
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
    ///    `ThreadIndex` is unique, and threads cannot share `ThreadIndex`
    ///    values, so each thread accesses a different memory location.
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
    #[inline]
    pub fn get_mut<'kernel>(&mut self, idx: ThreadIndex<'kernel, IndexSpace>) -> Option<&mut T> {
        let i = idx.get();
        if i < self.len {
            // SAFETY:
            // - Bounds check passed above.
            // - `idx` is a ThreadIndex derived from hardware built-in variables,
            //   guaranteeing a unique index per thread (no data races).
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
    pub fn get_mut_indexed<'kernel>(
        &mut self,
        scope: &'kernel KernelScope<'kernel>,
    ) -> Option<(&mut T, ThreadIndex<'kernel, IS>)> {
        let idx = IS::from_scope(scope)?;
        let i = idx.get();
        if i < self.len {
            // SAFETY:
            // - bounds check passed above
            // - idx is a thread-unique witness in IS, freshly minted from
            //   hardware special registers (no laundering — !Copy, !Send,
            //   `'kernel`-bound)
            // - DisjointSlice was constructed with valid memory
            //   (from_raw_parts safety)
            Some((unsafe { &mut *self.ptr.add(i) }, idx))
        } else {
            None
        }
    }
}

// SAFETY: DisjointSlice can be sent between threads because:
// - Each thread will access unique elements (guaranteed by ThreadIndex)
// - The pointer and length are just data, no thread affinity
// - T: Send means the elements themselves can be sent between threads
unsafe impl<'a, T: Send, IndexSpace> Send for DisjointSlice<'a, T, IndexSpace> {}

// DisjointSlice auto-trait summary:
//   Send: yes (explicit impl above, when T: Send)
//   Sync: NO (not implemented) — each GPU thread gets its own copy of the
//         struct, then uses its unique ThreadIndex to access a different
//         element. Sharing &DisjointSlice across threads would allow
//         multiple threads to call get_mut() on the same struct, which
//         would produce aliasing &mut T references — unsound.
