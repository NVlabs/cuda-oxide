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
//! 3. `index_1d`: the X-only formula is unique only for a 1D launch. A typed
//!    1D contract proves that shape; the legacy path checks all Y/Z grid and
//!    block dimensions at runtime and makes the witness invalid on mismatch.
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
    fn from_scope<'kernel, Domain, Coordinates>(
        scope: &'kernel KernelScope<'kernel, Domain, Coordinates>,
    ) -> Option<ThreadIndex<'kernel, Self>>
    where
        Domain: __internal::LaunchDomain;
}

impl IndexFormula for Index1D {
    #[inline(always)]
    fn from_scope<'kernel, Domain, Coordinates>(
        scope: &'kernel KernelScope<'kernel, Domain, Coordinates>,
    ) -> Option<ThreadIndex<'kernel, Self>>
    where
        Domain: __internal::LaunchDomain,
    {
        let index = __internal::index_1d(scope);
        index.is_valid().then_some(index)
    }
}

impl<const ROW_STRIDE: usize> IndexFormula for Index2D<ROW_STRIDE> {
    #[inline(always)]
    fn from_scope<'kernel, Domain, Coordinates>(
        scope: &'kernel KernelScope<'kernel, Domain, Coordinates>,
    ) -> Option<ThreadIndex<'kernel, Self>>
    where
        Domain: __internal::LaunchDomain,
    {
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
/// `__internal::index_*`.
///
/// `Domain` records which launch axes the host checked. `Coordinates` records
/// whether the host proved that each active axis fits in 32 bits. The proc
/// macros choose both markers from `#[launch_contract]`; safe user code cannot
/// construct a scope because all fields and its constructor are private.
#[doc(hidden)]
pub struct KernelScope<
    'kernel,
    Domain = __internal::UnknownDomain,
    Coordinates = __internal::NativeCoordinates,
> {
    _kernel: PhantomData<fn(&'kernel mut ()) -> &'kernel mut ()>,
    _domain: PhantomData<fn() -> Domain>,
    _coordinates: PhantomData<fn() -> Coordinates>,
    _not_send_sync: PhantomData<*mut ()>,
}

/// Borrowed kernel-scope proof with one lifetime shared by the reference and
/// the invariant scope value.
#[doc(hidden)]
pub type KernelScopeRef<'kernel, Domain, Coordinates> =
    &'kernel KernelScope<'kernel, Domain, Coordinates>;

impl<'kernel, Domain, Coordinates> KernelScope<'kernel, Domain, Coordinates> {
    #[inline(always)]
    unsafe fn new() -> Self {
        KernelScope {
            _kernel: PhantomData,
            _domain: PhantomData,
            _coordinates: PhantomData,
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
/// The proof-carrying [`index_1d32`] and [`coord_2d32`] functions are not on
/// this list. They take the capability named explicitly by
/// `#[kernel(scope = ...)]`, so ordinary Rust aliases and same-named local
/// functions behave normally.
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
    unsafe fn new<Domain, Coordinates>(
        raw: usize,
        valid: bool,
        _scope: &'kernel KernelScope<'kernel, Domain, Coordinates>,
    ) -> Self {
        ThreadIndex {
            // Keep ThreadIndex a scalar/newtype for the device ABI. usize::MAX
            // is reserved as the invalid legacy witness; a mathematically
            // valid 2D index at that one value is conservatively rejected.
            raw: if valid && raw != usize::MAX {
                raw
            } else {
                usize::MAX
            },
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

    /// Whether the launch shape and index arithmetic produced a valid witness.
    ///
    /// Legacy uncontracted kernels learn their rank only at runtime. A 1D
    /// formula under a 2D launch therefore yields an invalid witness instead
    /// of allowing two threads to alias the same output element.
    #[inline(always)]
    pub fn is_valid(&self) -> bool {
        self.raw != usize::MAX
    }

    /// Check if this index is less than a bound.
    ///
    /// Convenience method for bounds checking.
    #[inline(always)]
    pub fn in_bounds(&self, len: usize) -> bool {
        self.is_valid() && self.raw < len
    }
}

impl<'kernel, IndexSpace> fmt::Debug for ThreadIndex<'kernel, IndexSpace> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadIndex")
            .field("raw", &self.raw)
            .field("valid", &self.is_valid())
            .finish()
    }
}

/// A 32-bit, thread-unique index for a validated 1D launch.
///
/// This witness is available only in a kernel with
/// `#[launch_contract(domain = 1, coordinates = u32)]`. The host validates
/// `grid.x * block.x <= 2^32` before launch, so the index calculation and all
/// later tile-offset calculations can stay in `u32`.
///
/// Like [`ThreadIndex`], it is deliberately `!Copy`, `!Clone`, `!Send`, and
/// `!Sync`.
pub struct ThreadIndex32<'kernel> {
    raw: u32,
    _kernel: PhantomData<fn(&'kernel mut ()) -> &'kernel mut ()>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'kernel> ThreadIndex32<'kernel> {
    #[inline(always)]
    unsafe fn new(
        raw: u32,
        _scope: &'kernel KernelScope<'kernel, __internal::Domain1, __internal::U32Coordinates>,
    ) -> Self {
        Self {
            raw,
            _kernel: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Return the validated 32-bit global thread index.
    #[inline(always)]
    pub fn get(&self) -> u32 {
        self.raw
    }
}

impl fmt::Debug for ThreadIndex32<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ThreadIndex32").field(&self.raw).finish()
    }
}

/// A pair of 32-bit global row/column coordinates for a validated 2D launch.
///
/// This witness is available only in a kernel with
/// `#[launch_contract(domain = 2, coordinates = u32)]`. It is packed into one
/// `u64`, preserving a simple scalar device layout while keeping both
/// coordinate calculations in `u32`.
///
/// `ThreadCoord2D32` is deliberately `!Copy`, `!Clone`, `!Send`, and `!Sync`.
pub struct ThreadCoord2D32<'kernel> {
    packed: u64,
    _kernel: PhantomData<fn(&'kernel mut ()) -> &'kernel mut ()>,
    _not_send_sync: PhantomData<*mut ()>,
}

impl<'kernel> ThreadCoord2D32<'kernel> {
    #[inline(always)]
    unsafe fn new(
        row: u32,
        col: u32,
        _scope: &'kernel KernelScope<'kernel, __internal::Domain2, __internal::U32Coordinates>,
    ) -> Self {
        Self {
            packed: ((row as u64) << 32) | col as u64,
            _kernel: PhantomData,
            _not_send_sync: PhantomData,
        }
    }

    /// Global row coordinate (`blockIdx.y * blockDim.y + threadIdx.y`).
    #[inline(always)]
    pub fn row(&self) -> u32 {
        (self.packed >> 32) as u32
    }

    /// Global column coordinate (`blockIdx.x * blockDim.x + threadIdx.x`).
    #[inline(always)]
    pub fn col(&self) -> u32 {
        self.packed as u32
    }
}

impl fmt::Debug for ThreadCoord2D32<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadCoord2D32")
            .field("row", &self.row())
            .field("col", &self.col())
            .finish()
    }
}

#[doc(hidden)]
pub mod __internal {
    use super::{
        Index1D, Index2D, KernelScope, Runtime2DIndex, ThreadCoord2D32, ThreadIndex, ThreadIndex32,
    };

    mod sealed {
        pub trait Sealed {}
    }

    /// Scope marker for kernels without a typed launch-domain contract.
    pub enum UnknownDomain {}
    /// Scope marker for host-validated 1D launches.
    pub enum Domain1 {}
    /// Scope marker for host-validated 2D launches.
    pub enum Domain2 {}
    /// Scope marker for host-validated 3D launches.
    pub enum Domain3 {}

    /// Scope marker for ordinary target-width coordinate arithmetic.
    pub enum NativeCoordinates {}
    /// Scope marker for launches whose active coordinate products fit `u32`.
    pub enum U32Coordinates {}

    impl sealed::Sealed for UnknownDomain {}
    impl sealed::Sealed for Domain1 {}
    impl sealed::Sealed for Domain2 {}
    impl sealed::Sealed for Domain3 {}

    /// Internal facts carried by launch-domain marker types.
    ///
    /// This trait is sealed. It is public only because proc-macro-generated
    /// code mentions the marker types in public kernel bodies.
    #[doc(hidden)]
    pub trait LaunchDomain: sealed::Sealed {
        /// Highest active axis the launch contract permits (`0` = unknown).
        const MAX_DIMENSIONS: u8;
    }

    impl LaunchDomain for UnknownDomain {
        const MAX_DIMENSIONS: u8 = 0;
    }

    impl LaunchDomain for Domain1 {
        const MAX_DIMENSIONS: u8 = 1;
    }

    impl LaunchDomain for Domain2 {
        const MAX_DIMENSIONS: u8 = 2;
    }

    impl LaunchDomain for Domain3 {
        const MAX_DIMENSIONS: u8 = 3;
    }

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
    /// - **Plain `fn` device helpers (no annotation)** can't acquire a legacy
    ///   `ThreadIndex`. The public legacy `thread::index_*` items resolve fine
    ///   (they're `unreachable!` stubs), but without rewriting the stub body
    ///   executes and panics on first call.
    /// - **`#[device]` fns** can call the legacy `thread::index_1d` / 2D
    ///   helpers, but the returned
    ///   `ThreadIndex<'kernel, _>` borrows from the helper's local scope —
    ///   you can use it inside the helper, you can't return it out.
    ///   `#[device]` is mainly for FFI exports via LTOIR, where this
    ///   doesn't bite in practice. They cannot call `index_1d32`: only a
    ///   prepared kernel entry owns its typed launch proof, so helpers receive
    ///   the index or a checked view as an argument instead.
    ///   Contract-backed [`ThreadIndex32`] is different: a device helper cannot
    ///   create the host launch proof, so its caller must pass an index or checked
    ///   view into the helper.
    #[inline(always)]
    pub unsafe fn make_kernel_scope<'kernel, Domain, Coordinates>()
    -> KernelScope<'kernel, Domain, Coordinates> {
        unsafe { KernelScope::new() }
    }

    #[inline(always)]
    fn checked_axis(block: u32, block_size: u32, thread: u32) -> usize {
        let block = block as usize;
        let block_size = block_size as usize;
        let thread = thread as usize;
        if block_size == 0 || block > (usize::MAX - thread) / block_size {
            usize::MAX
        } else {
            block * block_size + thread
        }
    }

    #[inline(always)]
    fn one_dimensional_launch<Domain: LaunchDomain, Coordinates>(
        _scope: &KernelScope<'_, Domain, Coordinates>,
    ) -> bool {
        let y_is_unit = if Domain::MAX_DIMENSIONS == 1 {
            true
        } else {
            super::blockDim_y() == 1 && super::gridDim_y() == 1
        };
        let z_is_unit = if Domain::MAX_DIMENSIONS == 1 || Domain::MAX_DIMENSIONS == 2 {
            true
        } else {
            super::blockDim_z() == 1 && super::gridDim_z() == 1
        };
        y_is_unit && z_is_unit
    }

    #[inline(always)]
    fn at_most_two_dimensional_launch<Domain: LaunchDomain, Coordinates>(
        _scope: &KernelScope<'_, Domain, Coordinates>,
    ) -> bool {
        if Domain::MAX_DIMENSIONS == 1 || Domain::MAX_DIMENSIONS == 2 {
            true
        } else {
            super::blockDim_z() == 1 && super::gridDim_z() == 1
        }
    }

    /// Real `index_1d` intrinsic the `#[kernel]` / `#[device]` macros call in
    /// place of the public `super::index_1d` stub. Returns
    /// `blockIdx.x * blockDim.x + threadIdx.x`.
    ///
    /// The X-only formula is unique for a 1D launch. A typed `Domain1` scope
    /// proves that shape without device guards; other scopes validate every
    /// Y/Z grid and block dimension and return an invalid sentinel on mismatch.
    #[inline(always)]
    pub fn index_1d<'kernel>(
        scope: &'kernel KernelScope<'kernel, impl LaunchDomain, impl Sized>,
    ) -> ThreadIndex<'kernel, Index1D> {
        let raw = checked_axis(
            super::blockIdx_x(),
            super::blockDim_x(),
            super::threadIdx_x(),
        );
        let valid = one_dimensional_launch(scope) && raw != usize::MAX;
        unsafe { ThreadIndex::new(raw, valid, scope) }
    }

    /// Real `index_2d::<ROW_STRIDE>` intrinsic the macros call in place of the
    /// public `super::index_2d` stub. `Some(row * ROW_STRIDE + col)` when
    /// `col < ROW_STRIDE`, else `None`. Unique per thread for a 2D launch
    /// (`blockDim.z == gridDim.z == 1`); the const stride is in the witness type.
    #[inline(always)]
    pub fn index_2d<'kernel, const ROW_STRIDE: usize>(
        scope: &'kernel KernelScope<'kernel, impl LaunchDomain, impl Sized>,
    ) -> Option<ThreadIndex<'kernel, Index2D<ROW_STRIDE>>> {
        if !at_most_two_dimensional_launch(scope) {
            return None;
        }
        let row = checked_axis(
            super::blockIdx_y(),
            super::blockDim_y(),
            super::threadIdx_y(),
        );
        let col = checked_axis(
            super::blockIdx_x(),
            super::blockDim_x(),
            super::threadIdx_x(),
        );
        if row == usize::MAX || col == usize::MAX || col >= ROW_STRIDE {
            return None;
        }
        // col < ROW_STRIDE proves a non-zero divisor. Check the complete
        // linear expression before performing either operation.
        if row > (usize::MAX - col) / ROW_STRIDE {
            return None;
        }
        let raw = row * ROW_STRIDE + col;
        Some(unsafe { ThreadIndex::new(raw, true, scope) })
    }

    /// Real `index_2d_runtime` intrinsic the macros call in place of the public
    /// `super::index_2d_runtime` stub. Like `index_2d` but the row stride is a
    /// runtime value, so cross-thread uniqueness is the caller's `unsafe`
    /// obligation (every thread must pass the same `row_stride`).
    #[inline(always)]
    pub unsafe fn index_2d_runtime<'kernel>(
        scope: &'kernel KernelScope<'kernel, impl LaunchDomain, impl Sized>,
        row_stride: usize,
    ) -> Option<ThreadIndex<'kernel, Runtime2DIndex>> {
        if !at_most_two_dimensional_launch(scope) {
            return None;
        }
        let row = checked_axis(
            super::blockIdx_y(),
            super::blockDim_y(),
            super::threadIdx_y(),
        );
        let col = checked_axis(
            super::blockIdx_x(),
            super::blockDim_x(),
            super::threadIdx_x(),
        );
        if row == usize::MAX || col == usize::MAX || col >= row_stride {
            return None;
        }
        if row > (usize::MAX - col) / row_stride {
            return None;
        }
        let raw = row * row_stride + col;
        Some(unsafe { ThreadIndex::new(raw, true, scope) })
    }

    /// 32-bit index intrinsic for an exactly typed 1D launch scope.
    ///
    /// Host preparation proved `grid.x * block.x <= 2^32`. Therefore the
    /// mathematical result fits in `u32`; wrapping operations state that proof
    /// directly and avoid target-width promotion.
    #[inline(always)]
    pub fn index_1d32<'kernel>(
        scope: &'kernel KernelScope<'kernel, Domain1, U32Coordinates>,
    ) -> ThreadIndex32<'kernel> {
        let raw = super::blockIdx_x()
            .wrapping_mul(super::blockDim_x())
            .wrapping_add(super::threadIdx_x());
        unsafe { ThreadIndex32::new(raw, scope) }
    }

    /// 32-bit row/column intrinsic for an exactly typed 2D launch scope.
    ///
    /// Host preparation proves that both active axis products are at most
    /// `2^32`. The wrapping operations therefore equal the mathematical
    /// results for every real thread while remaining narrow device arithmetic.
    #[inline(always)]
    pub fn coord_2d32<'kernel>(
        scope: &'kernel KernelScope<'kernel, Domain2, U32Coordinates>,
    ) -> ThreadCoord2D32<'kernel> {
        let row = super::blockIdx_y()
            .wrapping_mul(super::blockDim_y())
            .wrapping_add(super::threadIdx_y());
        let col = super::blockIdx_x()
            .wrapping_mul(super::blockDim_x())
            .wrapping_add(super::threadIdx_x());
        unsafe { ThreadCoord2D32::new(row, col, scope) }
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
/// This reads only the X dimension, so the formula requires
/// `blockDim.y == blockDim.z == gridDim.y == gridDim.z == 1`. Contracted 1D
/// kernels receive that proof from host preparation. Legacy kernels check the
/// four dimensions on-device; a mismatch creates an invalid witness, and safe
/// `DisjointSlice` access returns `None`.
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

/// Get the global 1D thread index using only 32-bit arithmetic.
///
/// This is the fast, safe counterpart to [`index_1d`] for kernels declared
/// with `#[launch_contract(domain = 1, coordinates = u32)]`. Host-side launch
/// preparation rejects shapes where `grid.x * block.x > 2^32`, so every
/// produced coordinate fits exactly in a `u32`.
///
/// ```rust,ignore
/// #[kernel(scope = scope)]
/// #[launch_contract(domain = 1, coordinates = u32)]
/// fn vector(mut out: DisjointSlice<u32>) {
///     let thread = thread::index_1d32(scope);
///     if let Some(mut element) = out.element_thread32(thread) {
///         element.write(7);
///     }
/// }
/// ```
#[inline(always)]
pub fn index_1d32<'kernel>(
    scope: KernelScopeRef<'kernel, __internal::Domain1, __internal::U32Coordinates>,
) -> ThreadIndex32<'kernel> {
    __internal::index_1d32(scope)
}

/// Get global 2D row/column coordinates using 32-bit arithmetic.
///
/// This is available only to kernels declared with
/// `#[launch_contract(domain = 2, coordinates = u32)]`. The explicit scope
/// argument is created only by a matching `#[kernel(scope = ...)]` entry.
///
/// ```rust,ignore
/// #[kernel(scope = scope)]
/// #[launch_contract(domain = 2, coordinates = u32, block = (16, 16, 1))]
/// fn epilogue(mut out: DisjointSlice<f32, RowMajorTiles<2, 4, 4096>>) {
///     if let Some(mut tile) = out.tile_2d32(thread::coord_2d32(scope)) {
///         tile.at_const::<0, 0>().write(1.0);
///     }
/// }
/// ```
#[inline(always)]
pub fn coord_2d32<'kernel>(
    scope: KernelScopeRef<'kernel, __internal::Domain2, __internal::U32Coordinates>,
) -> ThreadCoord2D32<'kernel> {
    __internal::coord_2d32(scope)
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
/// This ignores the Z dimension, so it also requires
/// `blockDim.z == gridDim.z == 1`. Typed 1D/2D contracts prove that condition;
/// other paths check it on-device and return `None` on mismatch.
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
    (blockIdx_y() as usize) * (blockDim_y() as usize) + (threadIdx_y() as usize)
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
// SM and Grid Identification
// =============================================================================

/// Sample the current SM (streaming multiprocessor) identifier.
///
/// Returns the `%smid` special register: the SM on which this thread is
/// executing at the moment of the read. The value may change if the thread is
/// rescheduled after preemption, so use this for profiling and diagnostics,
/// not as a stable work-partitioning key.
///
/// Values are below [`nsmid()`], but SM identifiers need not be contiguous.
///
/// # PTX
///
/// `mov.u32 %r, %smid;`
#[inline(never)]
pub fn smid() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.smid()
    unreachable!("smid called outside CUDA kernel context")
}

/// Read the maximum SM ID + 1 (number of SM slots).
///
/// Returns the `%nsmid` special register. Note that SM IDs may not be
/// contiguous, so this is the upper bound, not the count of active SMs.
///
/// # PTX
///
/// `mov.u32 %r, %nsmid;`
#[inline(never)]
pub fn nsmid() -> u32 {
    // Lowered to: call i32 @llvm.nvvm.read.ptx.sreg.nsmid()
    unreachable!("nsmid called outside CUDA kernel context")
}

/// Read the grid's temporal launch identifier.
///
/// Returns the full 64-bit `%gridid` special register. CUDA debuggers use this
/// per-grid value to distinguish CTAs and clusters in concurrently executing
/// grids and across repeated launches.
///
/// # PTX
///
/// `mov.u64 %rd, %gridid;`
#[inline(never)]
pub fn gridid() -> u64 {
    // Lowered to exact inline PTX because LLVM's intrinsic exposes only the
    // legacy low-32-bit form.
    unreachable!("gridid called outside CUDA kernel context")
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

/// Compiler marker for a typed launch domain and coordinate width.
///
/// `#[launch_contract]` inserts this call. The MIR importer removes it before
/// code generation, so it emits no device instructions.
///
/// # Safety
///
/// `DOMAIN` must describe the launch shape enforced by the host contract. When
/// `U32_COORDINATES` is true, host preparation must also prove that the product
/// of grid and block extents on every active axis is at most `2^32`. Kernel
/// authors should use `#[launch_contract]` instead of calling this marker.
#[doc(hidden)]
#[inline(never)]
pub unsafe fn __launch_contract_config<const DOMAIN: u8, const U32_COORDINATES: bool>() {
    // Compiler marker: deliberately empty and removed during MIR import.
}

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
