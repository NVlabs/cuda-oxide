/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(non_snake_case)]
//! CUDA thread intrinsics and thread-safe index types.
//!
//! This module provides:
//! - `ThreadIndex<'kernel, IndexSpace>`: a typed witness derived from
//!   hardware built-in variables, with a `'kernel` lifetime that pins it
//!   to the kernel body
//! - Thread intrinsics: `threadIdx_x`, `blockIdx_x`, etc.
//! - Index helpers: `index_1d`, `index_2d::<S>`, `unsafe index_2d_runtime`
//!   that return typed `ThreadIndex` witnesses
//! - `IndexFormula`: a marker trait for index spaces that can be derived
//!   from the kernel scope alone (used by `DisjointSlice::get_mut_indexed`)
//!
//! # Safety Model
//!
//! The safety of parallel writes to `DisjointSlice` relies on each thread
//! accessing a unique memory location. This is guaranteed as follows:
//!
//! 1. **ThreadIndex** can only be constructed by trusted functions:
//!    `index_1d()`, `index_2d::<S>()`, and the unsafe `index_2d_runtime(s)`.
//! 2. These functions derive the index from hardware built-in variables
//!    (`threadIdx`, `blockIdx`, `blockDim`) -- read-only special registers
//!    assigned by the runtime at kernel launch. The formula
//!    `outer * stride + inner` combines these into a scalar index per thread.
//! 3. `index_1d`: unique per thread only for a 1D launch
//!    (`blockDim.y == blockDim.z == 1` and `gridDim.y == gridDim.z == 1`). It
//!    reads only the X registers, so a 2D/3D launch collides; see issue #115.
//! 4. `index_2d::<S>()`: unique per thread for const-stride 2D grids.
//!    The stride lives in the witness type, and `DisjointSlice` only
//!    accepts indices from the matching index space -- mismatched
//!    strides are a compile error.
//! 5. `unsafe index_2d_runtime(s)`: caller asserts every thread used the
//!    same `s`. The type system can't prove uniformity for runtime
//!    strides; the `unsafe` keyword is the contract.
//! 6. The witness is `!Send + !Sync + !Copy + !Clone` and `'kernel`-scoped,
//!    so threads can't launder it through shared memory and it can't
//!    outlive the kernel body.

use core::fmt;
use core::marker::PhantomData;

// =============================================================================
// ThreadIndex - Type-Safe Thread-Unique Index
// =============================================================================

/// Type-level index space for the standard 1D index formula.
pub enum Index1D {}

/// Type-level index space for a 2D row-major index with a const row stride.
pub enum Index2D<const ROW_STRIDE: usize> {}

/// Index spaces whose `ThreadIndex` can be derived from the kernel scope alone.
///
/// `Index1D` and `Index2D<S>` impl this — their formulas take no runtime
/// inputs. [`Runtime2DIndex`] does **not** impl this, because the row stride
/// is a runtime value the type system can't see; reach for the unsafe
/// [`index_2d_runtime`] when you need a runtime stride.
///
/// Used by `DisjointSlice::get_mut_indexed` to mint the per-thread index
/// in the same call that resolves it to a mutable reference.
pub trait IndexFormula: Sized {
    #[doc(hidden)]
    fn from_scope<'kernel>(
        scope: &'kernel KernelScope<'kernel>,
    ) -> Option<ThreadIndex<'kernel, Self>>;
}

impl IndexFormula for Index1D {
    #[inline(always)]
    fn from_scope<'kernel>(
        scope: &'kernel KernelScope<'kernel>,
    ) -> Option<ThreadIndex<'kernel, Self>> {
        Some(__internal::index_1d(scope))
    }
}

impl<const ROW_STRIDE: usize> IndexFormula for Index2D<ROW_STRIDE> {
    #[inline(always)]
    fn from_scope<'kernel>(
        scope: &'kernel KernelScope<'kernel>,
    ) -> Option<ThreadIndex<'kernel, Self>> {
        __internal::index_2d::<ROW_STRIDE>(scope)
    }
}

/// Type-level index space for manually audited runtime-stride 2D indexing.
///
/// Two `ThreadIndex<'_, Runtime2DIndex>` values produced under different runtime
/// strides have the same type, so the type system can't tell them apart. The
/// `unsafe` on [`index_2d_runtime`] is the only thing keeping callers honest:
/// every thread in the kernel that feeds a `Runtime2DIndex` into the same
/// `DisjointSlice` must have used the same `row_stride`. If you can pin the
/// stride at compile time, prefer [`index_2d`] — the const-generic version
/// makes a stride mismatch a type error instead of a contract.
pub enum Runtime2DIndex {}

/// Stack-local witness produced by `make_kernel_scope` and consumed by
/// `__internal::index_*`. The `'kernel` lifetime tags every `ThreadIndex`
/// minted from it. Hidden from public docs because users never name this
/// type — the macros inject it.
#[doc(hidden)]
pub struct KernelScope<'kernel> {
    _kernel: PhantomData<fn(&'kernel mut ()) -> &'kernel mut ()>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'kernel> KernelScope<'kernel> {
    #[inline(always)]
    unsafe fn new() -> Self {
        KernelScope {
            _kernel: PhantomData,
            _not_send_sync: PhantomData,
        }
    }
}

/// A thread-unique index derived from hardware built-in variables (special registers).
///
/// `ThreadIndex` cannot be constructed directly. The contained `usize` is
/// unique per thread, which is what makes parallel writes to `DisjointSlice`
/// race-free without synchronisation.
///
/// The index-space parameter ties each witness to the indexing scheme that
/// created it. A `DisjointSlice<T, Index2D<128>>` won't accept a
/// `ThreadIndex<'_, Index2D<256>>`, so mixing 2D strides is rejected at
/// compile time.
///
/// `ThreadIndex` is intentionally `!Send`, `!Sync`, `!Copy`, and `!Clone`,
/// so safe code can't duplicate a witness or smuggle one to a different
/// thread. The `'kernel` lifetime is borrowed from a stack-local scope the
/// proc macros inject; it can't outlive the kernel body.
///
/// # Construction
///
/// `ThreadIndex` cannot be constructed directly. Use one of the trusted
/// functions:
/// - [`index_1d()`] — for 1D grids
/// - [`index_2d()`] — for const-stride 2D grids
/// - [`index_2d_runtime()`] — unsafe runtime-stride escape hatch
///
/// # Where you can call them
///
/// The `'kernel` scope only exists inside `#[kernel]` and `#[device]`
/// bodies, which has two practical consequences:
///
/// - **Plain `fn` device helpers (no annotation)** can't acquire a
///   `ThreadIndex`. The public `thread::index_*` items are `unreachable!`
///   stubs — they compile and import fine, but calling one outside an
///   annotated body panics on first call. The macros rewrite real
///   call sites to `thread::__internal::*`, which is what actually runs
///   on the device. If a helper needs an index, take it as a parameter.
/// - **`#[device]` fns** *can* call `thread::index_*`, but they can't
///   return the resulting `ThreadIndex` — `'kernel` is borrowed from the
///   helper's local scope and dies at function exit. Use the witness
///   inside the helper. (`#[device]` is mainly for FFI exports via
///   LTOIR, where this restriction doesn't bite.)
///
/// # Reserved names inside `#[kernel]` and `#[device]`
///
/// The macros rewrite a small set of names inside annotated bodies so
/// the user never has to thread the kernel scope through by hand:
///
/// - free functions: `index_1d`, `index_2d`, `index_2d_runtime`
/// - methods (zero-arg call sites): `get_mut_indexed`
///
/// Free-function calls are matched on path tail, so all of these resolve
/// to the same intrinsic:
///
/// ```rust,ignore
/// thread::index_1d()
/// cuda_device::thread::index_1d()
/// use cuda_device::thread::index_1d;  index_1d()
/// use cuda_device::thread::index_1d as get_idx;  get_idx() // not rewritten — alias
/// ```
///
/// Method calls are matched on the method name only — `slice.get_mut_indexed()`
/// has the kernel scope spliced in as the (currently invisible)
/// `&KernelScope` argument the method actually takes.
///
/// The trade-off: if you define a local `fn index_1d` (or any of the
/// other reserved names) and call it from inside `#[kernel]` or
/// `#[device]`, the macro will silently rewrite that call too. Pick a
/// different name (e.g. `compute_index_1d`, `pop_indexed`) for any
/// helper you want to keep.
///
/// # Example
///
/// ```rust,ignore
/// use cuda_device::{DisjointSlice, kernel, thread};
///
/// #[kernel]
/// fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
///     let idx = thread::index_1d();
///     let i = idx.get();
///     if let Some(c_elem) = c.get_mut(idx) {
///         *c_elem = a[i] + b[i];
///     }
/// }
/// ```
pub struct ThreadIndex<'kernel, IndexSpace = Index1D> {
    raw: usize,
    _kernel: PhantomData<fn(&'kernel mut ()) -> &'kernel mut ()>,
    _space: PhantomData<fn() -> IndexSpace>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'kernel, IndexSpace> ThreadIndex<'kernel, IndexSpace> {
    #[inline(always)]
    unsafe fn new(raw: usize, _scope: &'kernel KernelScope<'kernel>) -> Self {
        ThreadIndex {
            raw,
            _kernel: PhantomData,
            _space: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Get the raw index value.
    ///
    /// Use this when you need the index for array indexing on regular slices.
    #[inline(always)]
    pub fn get(&self) -> usize {
        self.raw
    }

    /// Check if this index is less than a bound.
    ///
    /// Convenience method for bounds checking.
    #[inline(always)]
    pub fn in_bounds(&self, len: usize) -> bool {
        self.raw < len
    }
}

impl<'kernel, IndexSpace> fmt::Debug for ThreadIndex<'kernel, IndexSpace> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ThreadIndex").field(&self.raw).finish()
    }
}

#[doc(hidden)]
pub mod __internal {
    use super::{Index1D, Index2D, KernelScope, Runtime2DIndex, ThreadIndex};

    /// Mints a fresh `KernelScope` whose `'kernel` lifetime backs every
    /// `ThreadIndex` produced inside this kernel/device body.
    ///
    /// # Safety
    ///
    /// Only the `#[kernel]` and `#[device]` proc macros may call this. They
    /// inject exactly one call at the top of the rewritten function body and
    /// bind the result to a stack local, so the lifetime can't escape.
    /// Calling it anywhere else lets the caller forge `ThreadIndex` values
    /// via `__internal::index_*`, which breaks the entire safety story.
    ///
    /// # Call-context consequences
    ///
    /// The "only macros call this" rule shapes where `thread::index_*` is
    /// usable:
    ///
    /// - **Plain `fn` device helpers (no annotation)** can't acquire a
    ///   `ThreadIndex`. The public `thread::index_*` items resolve fine
    ///   (they're `unreachable!` stubs), so the call compiles, but
    ///   without the macro rewriting it to the `__internal::*` form
    ///   the stub body executes and panics on first call.
    /// - **`#[device]` fns** *can* call `thread::index_*`, but the returned
    ///   `ThreadIndex<'kernel, _>` borrows from the helper's local scope —
    ///   you can use it inside the helper, you can't return it out.
    ///   `#[device]` is mainly for FFI exports via LTOIR, where this
    ///   doesn't bite in practice.
    #[inline(always)]
    pub unsafe fn make_kernel_scope<'kernel>() -> KernelScope<'kernel> {
        unsafe { KernelScope::new() }
    }

    /// Real `index_1d` intrinsic the `#[kernel]` / `#[device]` macros call in
    /// place of the public `super::index_1d` stub. Returns
    /// `blockIdx.x * blockDim.x + threadIdx.x`.
    ///
    /// Unique per thread **only for a 1D launch** (`blockDim.y == blockDim.z ==
    /// 1` and `gridDim.y == gridDim.z == 1`); a 2D/3D launch collides because
    /// this reads only the X registers. Tracked in issue #115.
    #[inline(always)]
    pub fn index_1d<'kernel>(
        scope: &'kernel KernelScope<'kernel>,
    ) -> ThreadIndex<'kernel, Index1D> {
        let tid = super::threadIdx_x() as usize;
        let bid = super::blockIdx_x() as usize;
        let bdim = super::blockDim_x() as usize;
        unsafe { ThreadIndex::new(bid * bdim + tid, scope) }
    }

    /// Real `index_2d::<ROW_STRIDE>` intrinsic the macros call in place of the
    /// public `super::index_2d` stub. `Some(row * ROW_STRIDE + col)` when
    /// `col < ROW_STRIDE`, else `None`. Unique per thread for a 2D launch
    /// (`blockDim.z == gridDim.z == 1`); the const stride is in the witness type.
    #[inline(always)]
    pub fn index_2d<'kernel, const ROW_STRIDE: usize>(
        scope: &'kernel KernelScope<'kernel>,
    ) -> Option<ThreadIndex<'kernel, Index2D<ROW_STRIDE>>> {
        let row = (super::blockIdx_y() as usize) * (super::blockDim_y() as usize)
            + (super::threadIdx_y() as usize);
        let col = (super::blockIdx_x() as usize) * (super::blockDim_x() as usize)
            + (super::threadIdx_x() as usize);
        if col < ROW_STRIDE {
            Some(unsafe { ThreadIndex::new(row * ROW_STRIDE + col, scope) })
        } else {
            None
        }
    }

    /// Real `index_2d_runtime` intrinsic the macros call in place of the public
    /// `super::index_2d_runtime` stub. Like `index_2d` but the row stride is a
    /// runtime value, so cross-thread uniqueness is the caller's `unsafe`
    /// obligation (every thread must pass the same `row_stride`).
    #[inline(always)]
    pub unsafe fn index_2d_runtime<'kernel>(
        scope: &'kernel KernelScope<'kernel>,
        row_stride: usize,
    ) -> Option<ThreadIndex<'kernel, Runtime2DIndex>> {
        let row = (super::blockIdx_y() as usize) * (super::blockDim_y() as usize)
            + (super::threadIdx_y() as usize);
        let col = (super::blockIdx_x() as usize) * (super::blockDim_x() as usize)
            + (super::threadIdx_x() as usize);
        if col < row_stride {
            Some(unsafe { ThreadIndex::new(row * row_stride + col, scope) })
        } else {
            None
        }
    }
}

// =============================================================================
// 1D Index Helper
// =============================================================================

/// Get the global 1D thread index.
///
/// Computes: `blockIdx.x * blockDim.x + threadIdx.x`
///
/// Designed for **1D launches** (only the X dimension is used). For 2D grids
/// use [`index_2d`] instead.
///
/// # Uniqueness
///
/// This reads only the X dimension, so the index is unique per thread **only
/// when the launch is 1D**: `blockDim.y == blockDim.z == 1` and
/// `gridDim.y == gridDim.z == 1`.
///
/// Under a 2D or 3D launch, threads that share the same X but differ in Y or Z
/// get the *same* index, which would alias the same `DisjointSlice` slot. Every
/// shipped example launches 1D, so nothing hits this today. Tracked in issue
/// #115 (a fix that makes this sound under any launch is being weighed).
///
/// # Example
///
/// ```rust,ignore
/// let idx = index_1d();
/// let i = idx.get();
/// if let Some(c_elem) = c.get_mut(idx) {
///     *c_elem = a[i] + b[i];
/// }
/// ```
///
/// # Stub body
///
/// Calls inside `#[kernel]` / `#[device]` are rewritten by the macros
/// to the real intrinsic path (`thread::__internal::index_1d`). The
/// public function exists only so imports and aliases resolve cleanly;
/// invoking it directly from host code panics.
#[inline(always)]
pub fn index_1d<'kernel>() -> ThreadIndex<'kernel> {
    unreachable!(
        "thread::index_1d called outside #[kernel] / #[device] — the macro rewrites real call sites; the public item is a stub"
    )
}

// =============================================================================
// 2D Index Helper
// =============================================================================

/// Get the global 2D thread index for a const row stride, linearized to 1D.
///
/// Returns `Some(ThreadIndex)` when `col < ROW_STRIDE`, `None` otherwise.
///
/// Computes: `row * ROW_STRIDE + col`
///
/// Where:
/// - `row = blockIdx.y * blockDim.y + threadIdx.y`
/// - `col = blockIdx.x * blockDim.x + threadIdx.x`
///
/// # Why the stride is const-generic
///
/// The row stride is part of the returned witness type:
/// `ThreadIndex<Index2D<ROW_STRIDE>>`. A `DisjointSlice` in a different domain
/// will not accept it, so accidentally mixing `index_2d::<100>()` and
/// `index_2d::<200>()` for the same output is a compile-time error.
///
/// # Uniqueness Guarantee
///
/// The formula `row * ROW_STRIDE + col` is injective when
/// `col < ROW_STRIDE`. The internal check returns `None` for threads where
/// `col >= ROW_STRIDE`, so the surviving `ThreadIndex` values are unique.
///
/// **Proof sketch (within one stride):** Two threads with distinct
/// `(row_a, col_a)` and `(row_b, col_b)` where both `col_a < stride` and
/// `col_b < stride`:
///
/// ```text
///   row_a * stride + col_a == row_b * stride + col_b
///   => (row_a - row_b) * stride == col_b - col_a
/// ```
///
/// `col_a, col_b ∈ [0, stride)`, so the RHS is in `(-stride, stride)`.
/// The LHS is a multiple of `stride`, so the only solution is
/// `row_a == row_b` AND `col_a == col_b`. Distinct hardware threads have
/// distinct `(row, col)` **for a 2D launch**.
///
/// This ignores the Z dimension, so it is unique only when
/// `blockDim.z == gridDim.z == 1`; a 3D launch would collide on Z. See issue
/// #115.
///
/// # Parameters
///
/// - `ROW_STRIDE`: The stride for row-major layout (typically the number
///   of columns `N`).
///
/// # Example
///
/// ```rust,ignore
/// // GEMM: C[row, col] = ...
/// let row = index_2d_row();
/// let col = index_2d_col();
/// if let Some(c_idx) = index_2d::<1024>() {
///     // col < 1024 is guaranteed by Some
///     if row < m {
///         if let Some(c_elem) = c.get_mut(c_idx) {
///             *c_elem = ...;
///         }
///     }
/// }
/// ```
///
/// # Stub body
///
/// Calls inside `#[kernel]` / `#[device]` are rewritten by the macros
/// to the real intrinsic path (`thread::__internal::index_2d::<ROW_STRIDE>`).
/// The public function exists only so imports and aliases resolve
/// cleanly; invoking it directly from host code panics.
#[inline(always)]
pub fn index_2d<'kernel, const ROW_STRIDE: usize>()
-> Option<ThreadIndex<'kernel, Index2D<ROW_STRIDE>>> {
    unreachable!(
        "thread::index_2d called outside #[kernel] / #[device] — the macro rewrites real call sites; the public item is a stub"
    )
}

/// Runtime-stride 2D indexing escape hatch.
///
/// # Safety
///
/// Every thread in the kernel that uses the resulting index with the same
/// `DisjointSlice<T, Runtime2DIndex>` must pass the same `row_stride`. Mixing
/// runtime strides can create colliding indices and data races.
///
/// # Stub body
///
/// Calls inside `#[kernel]` / `#[device]` are rewritten by the macros
/// to the real intrinsic path (`thread::__internal::index_2d_runtime`).
/// The public function exists only so imports and aliases resolve
/// cleanly; invoking it directly from host code panics.
#[inline(always)]
pub unsafe fn index_2d_runtime<'kernel>(
    row_stride: usize,
) -> Option<ThreadIndex<'kernel, Runtime2DIndex>> {
    let _ = row_stride;
    unreachable!(
        "thread::index_2d_runtime called outside #[kernel] / #[device] — the macro rewrites real call sites; the public item is a stub"
    )
}

/// Get the row component of a 2D thread index.
///
/// Computes: `blockIdx.y * blockDim.y + threadIdx.y`
#[inline(always)]
pub fn index_2d_row() -> usize {
    (blockIdx_y() * blockDim_y() + threadIdx_y()) as usize
}

/// Get the column component of a 2D thread index.
///
/// Computes: `blockIdx.x * blockDim.x + threadIdx.x`
#[inline(always)]
pub fn index_2d_col() -> usize {
    (blockIdx_x() as usize) * (blockDim_x() as usize) + (threadIdx_x() as usize)
}

// =============================================================================
// X-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.x (thread index within block, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn threadIdx_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
    unreachable!("threadIdx_x called outside CUDA kernel context")
}

/// Get blockIdx.x (block index within grid, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockIdx_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
    unreachable!("blockIdx_x called outside CUDA kernel context")
}

/// Get blockDim.x (block dimension, X dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockDim_x() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
    unreachable!("blockDim_x called outside CUDA kernel context")
}

// =============================================================================
// Y-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.y (thread index within block, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn threadIdx_y() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.tid.y()
    unreachable!("threadIdx_y called outside CUDA kernel context")
}

/// Get blockIdx.y (block index within grid, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockIdx_y() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()
    unreachable!("blockIdx_y called outside CUDA kernel context")
}

/// Get blockDim.y (block dimension, Y dimension)
///
/// This function is recognized by the cuda-oxide compiler and replaced
/// with the appropriate PTX intrinsic. The body should never execute.
#[inline(never)]
pub fn blockDim_y() -> u32 {
    unreachable!("blockDim_y called outside CUDA kernel context")
}

// =============================================================================
// Z-Dimension Intrinsics
// =============================================================================

/// Get threadIdx.z (thread index within block, Z dimension).
#[inline(never)]
pub fn threadIdx_z() -> u32 {
    unreachable!("threadIdx_z called outside CUDA kernel context")
}

/// Get blockIdx.z (block index within grid, Z dimension).
#[inline(never)]
pub fn blockIdx_z() -> u32 {
    unreachable!("blockIdx_z called outside CUDA kernel context")
}

/// Get blockDim.z (block dimension, Z dimension).
#[inline(never)]
pub fn blockDim_z() -> u32 {
    unreachable!("blockDim_z called outside CUDA kernel context")
}

// =============================================================================
// Grid Dimensions (gridDim)
// =============================================================================

/// Get gridDim.x — number of blocks along the X axis of the grid.
#[inline(never)]
pub fn gridDim_x() -> u32 {
    unreachable!("gridDim_x called outside CUDA kernel context")
}

/// Get gridDim.y — number of blocks along the Y axis of the grid.
#[inline(never)]
pub fn gridDim_y() -> u32 {
    unreachable!("gridDim_y called outside CUDA kernel context")
}

/// Get gridDim.z — number of blocks along the Z axis of the grid.
#[inline(never)]
pub fn gridDim_z() -> u32 {
    unreachable!("gridDim_z called outside CUDA kernel context")
}

// =============================================================================
// Synchronization Intrinsics
// =============================================================================

/// Block-level thread synchronization barrier.
///
/// All threads in a block must reach this barrier before any thread can proceed.
/// This is equivalent to `__syncthreads()` in CUDA C/C++.
///
/// # Usage
///
/// ```rust,ignore
/// use cuda_device::thread;
///
/// // Write to shared memory
/// shared_tile[tid] = value;
///
/// // Ensure all threads have written before any thread reads
/// thread::sync_threads();
///
/// // Now safe to read values written by other threads
/// let neighbor = shared_tile[other_tid];
/// ```
///
/// # Safety
///
/// - All threads in the block must reach the same barrier (no divergent barriers)
/// - Placing `sync_threads()` inside a conditional where not all threads enter
///   will cause deadlock
#[inline(never)]
pub fn sync_threads() {
    // Lowered to: call void @llvm.nvvm.barrier0()
    unreachable!("sync_threads called outside CUDA kernel context")
}

// =============================================================================
// Compile-Time Launch Bounds Configuration
// =============================================================================

/// Marker function for compile-time launch bounds configuration.
///
/// This is a compile-time configuration marker that tells the compiler to emit
/// `.maxntid` and `.minnctapersm` PTX directives for this kernel. It does NOT
/// generate any runtime code.
///
/// # Usage
///
/// This function should NOT be called directly. Use the `#[launch_bounds(max, min)]`
/// attribute macro instead, which injects this marker:
///
/// ```rust,ignore
/// #[kernel]
/// #[launch_bounds(256)]           // max 256 threads per block
/// pub fn my_kernel(output: DisjointSlice<f32>) { ... }
///
/// #[kernel]
/// #[launch_bounds(256, 2)]        // max 256 threads, min 2 blocks per SM
/// pub fn optimized_kernel(output: DisjointSlice<f32>) { ... }
/// ```
///
/// # How It Works
///
/// 1. The `#[launch_bounds]` macro injects `__launch_bounds_config::<MAX, MIN>()` at kernel start
/// 2. MIR importer detects this call and extracts the const generic parameters
/// 3. The marker call is NOT compiled - it's removed during compilation
/// 4. LLVM export emits `!nvvm.annotations` with `maxntid` and `minctasm` metadata
/// 5. LLVM NVPTX backend emits `.maxntid` and `.minnctapersm` in PTX
///
/// # PTX Output
///
/// ```ptx
/// .entry my_kernel .maxntid 256 .minnctapersm 2 { ... }
/// ```
///
/// # Parameters
///
/// - `MAX_THREADS` - Maximum threads per block (required). Maps to `.maxntid`.
/// - `MIN_BLOCKS` - Minimum blocks per SM for occupancy (optional, default 0 = unspecified).
///   Maps to `.minnctapersm`.
///
/// # Performance Impact
///
/// Launch bounds help the compiler:
/// - Allocate registers more efficiently
/// - Optimize occupancy (threads per SM)
/// - Make better scheduling decisions
///
/// Using appropriate launch bounds can significantly improve performance for
/// register-heavy kernels or kernels with specific occupancy requirements.
#[inline(never)]
pub fn __launch_bounds_config<const MAX_THREADS: u32, const MIN_BLOCKS: u32>() {
    // This function is detected at compile time and removed.
    // The const generics are extracted to set launch bounds.
    // No runtime code is generated.
}

/// Compile-time loop-unroll request marker (internal, do not call directly).
///
/// The `#[kernel]` and `#[device]` macros insert this marker at the start of an
/// annotated loop body. The MIR importer turns it into a `mir.unroll_hint`
/// operation, and the loop-unroll pass consumes that hint before lowering. It
/// generates no runtime code.
///
/// # Usage
///
/// Put the annotation directly on the loop. Bare `#[unroll]` requests full
/// unrolling; `#[unroll(N)]` requests `N` copies per trip:
///
/// ```rust,ignore
/// #[kernel]
/// pub fn my_kernel(mut output: DisjointSlice<u32>, n: u32) {
///     let tid = thread::index_1d();
///     if let Some(out_elem) = output.get_mut(tid) {
///         let mut sum = 0;
///         let mut i = 0;
///         #[unroll]
///         while i < 4 {
///             sum += i;
///             i += 1;
///         }
///         *out_elem = sum;
///     }
///
///     let mut i = 0;
///     #[unroll(4)]
///     while i < n {
///         i += 1;
///     }
/// }
/// ```
///
/// The pass currently recognizes explicit counted `while` loops. Range-based
/// `for` loops are not yet recognized.
///
/// Loops with several `continue` paths are supported. Full `#[unroll]` also
/// preserves `break` paths and multiple exit targets. Partial `#[unroll(N)]`
/// requires a positive counter step, a `<` or `<=` test, an unchanging limit,
/// and no exit besides the normal header test. Unsupported requests warn and
/// are not unrolled.
///
/// One annotation may create at most 1,024 body copies, 8,192 cloned basic
/// blocks, and 65,536 cloned operations. Larger requests warn and are not
/// unrolled.
///
/// # Parameters
///
/// - `FACTOR = 0` requests full unrolling of this loop and requires a
///   compile-time-known trip count.
/// - `FACTOR >= 2` requests partial unrolling of this loop by that factor.
#[inline(never)]
pub fn __unroll_config<const FACTOR: u32>() {
    // This function is detected at compile time and removed.
    // The const generic FACTOR is extracted to set the loop-unroll request.
    // No runtime code is generated.
}
