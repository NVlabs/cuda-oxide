/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Constant-memory support for CUDA kernels.
//!
//! [`ConstantMemory<T>`] is a wrapper for module-scope statics that live in CUDA
//! constant memory (PTX `.const`, address space 4). The host populates the
//! storage via `cuMemcpyHtoD`; device code reads it as if it were an
//! ordinary `static T`, returning a value-typed copy.
//!
//! # Usage
//!
//! Declare the static inside a `#[cuda_module]` and tag it with `#[constant]`:
//!
//! ```ignore
//! use cuda_device::{constant, cuda_module, kernel, thread, ConstantMemory, DisjointSlice};
//!
//! #[cuda_module]
//! mod kernels {
//!     use super::*;
//!
//!     #[constant]
//!     static COEFFS: ConstantMemory<[f32; 4]> = ConstantMemory::UNINIT;
//!
//!     #[kernel]
//!     pub fn apply(mut out: DisjointSlice<f32>) {
//!         let c = COEFFS.get();        // safe read; returns [f32; 4]
//!         let i = thread::index_1d().get();
//!         if let Some(e) = out.get_mut(thread::index_1d()) {
//!             *e = c[0] + c[1] * (i as f32);
//!         }
//!     }
//! }
//! ```
//!
//! Host code overrides the initializer between launches with the
//! macro-generated `set_<name>` methods on the loaded module:
//!
//! ```ignore
//! module.set_coeffs(&stream, &[10.0, 20.0, 30.0, 40.0])?;
//! ```
//!
//! # Why a wrapper type instead of a bare `static`
//!
//! A plain `static COEFFS: [f32; 4] = [1.0; 4];` would be constant-folded by
//! rustc — every read in device code is replaced with the literal initializer
//! values, and the host's `cuMemcpyHtoD` update becomes invisible. Wrapping
//! the storage in [`UnsafeCell`] prevents the fold by signalling interior
//! mutability, restoring the read-from-memory semantics that CUDA constant
//! memory requires.
//!
//! # Soundness: `Sync`
//!
//! Unlike [`SharedArray`](crate::SharedArray) (which is `!Sync` because
//! shared memory is per-block and requires barriers), `ConstantMemory<T>` is
//! `Sync`. CUDA constant memory has a single, host-controlled value visible
//! identically to every thread on the device, with no in-kernel writes; a
//! `&ConstantMemory<T>` from any thread is sound to read concurrently.

use core::cell::UnsafeCell;

/// Marker trait for values that may be stored in [`ConstantMemory`].
///
/// A constant-memory value is copied byte-for-byte from the host and may be
/// created as an all-zero placeholder by [`ConstantMemory::UNINIT`] before the
/// host populates it.
///
/// # Safety
///
/// Implementors must be safe to duplicate with a byte-for-byte copy, and the
/// all-zero bit pattern must be a valid value of `Self`. Do not implement this
/// trait for references, `NonZero*`, or any type containing a niche that makes
/// the all-zero bit pattern invalid. Custom structs should use an explicit
/// layout such as `#[repr(C)]` before opting in.
pub unsafe trait ConstantMemoryValue: Copy {}

macro_rules! impl_constant_memory_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl ConstantMemoryValue for $ty {}
        )+
    };
}

impl_constant_memory_value!(
    (),
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f16,
    f32,
    f64
);

unsafe impl<T: ConstantMemoryValue, const N: usize> ConstantMemoryValue for [T; N] {}
unsafe impl<T: ?Sized> ConstantMemoryValue for *const T {}
unsafe impl<T: ?Sized> ConstantMemoryValue for *mut T {}

macro_rules! impl_constant_memory_value_tuple {
    ($($name:ident),+ $(,)?) => {
        unsafe impl<$($name: ConstantMemoryValue),+> ConstantMemoryValue for ($($name,)+) {}
    };
}

impl_constant_memory_value_tuple!(A);
impl_constant_memory_value_tuple!(A, B);
impl_constant_memory_value_tuple!(A, B, C);
impl_constant_memory_value_tuple!(A, B, C, D);
impl_constant_memory_value_tuple!(A, B, C, D, E);
impl_constant_memory_value_tuple!(A, B, C, D, E, F);
impl_constant_memory_value_tuple!(A, B, C, D, E, F, G);
impl_constant_memory_value_tuple!(A, B, C, D, E, F, G, H);

/// A `static`-friendly wrapper that places `T` in CUDA constant memory
/// (`addrspace(4)`).
///
/// See the [module docs](self) for the full usage pattern.
#[repr(transparent)]
pub struct ConstantMemory<T: ConstantMemoryValue>(UnsafeCell<T>);

// SAFETY: ConstantMemory<T> is a host-populated, device-readonly cell. The host
// performs writes via `cuMemcpyHtoD` (synchronized with the calling thread);
// device code only reads. No concurrent writer exists on either side, so
// shared `&ConstantMemory<T>` across threads is sound.
unsafe impl<T: ConstantMemoryValue + Send> Sync for ConstantMemory<T> {}

impl<T: ConstantMemoryValue> ConstantMemory<T> {
    /// Placeholder value for a `#[constant]` static. The host must call
    /// the macro-generated `set_<name>` before any kernel reads the
    /// constant; honoring arbitrary non-zero initializers in codegen is
    /// not yet implemented.
    ///
    /// Mirrors the convention used by [`SharedArray::UNINIT`](crate::SharedArray)
    /// and [`Barrier::UNINIT`](crate::barrier::Barrier): a single placeholder
    /// constant for a type that's expected to be populated out-of-band.
    ///
    /// # Note
    ///
    /// The underlying bytes are zero. The [`ConstantMemoryValue`] bound is the
    /// safety gate that rules out types where the all-zero placeholder would
    /// violate Rust's validity invariants. Custom types can opt in with an
    /// `unsafe impl ConstantMemoryValue` when their layout and zero value make
    /// that promise true.
    #[allow(clippy::declare_interior_mutable_const)]
    pub const UNINIT: Self = ConstantMemory(UnsafeCell::new(unsafe {
        core::mem::MaybeUninit::<T>::zeroed().assume_init()
    }));

    /// Read the current value.
    ///
    /// Returns a by-value copy of the storage. Safe because constant memory
    /// is read-only from the device — there is no possibility of observing
    /// a torn write from another thread.
    ///
    /// The `UnsafeCell` interior prevents the compiler from hoisting reads
    /// across `set_<name>` boundaries, which means a `.get()` inside a hot
    /// loop will re-read on every iteration. For large `T` (e.g.
    /// `ConstantMemory<[f32; 1024]>`) call `.get()` once before the loop and
    /// reuse the local:
    ///
    /// ```ignore
    /// let coeffs = COEFFS.get();    // single load, hoisted by you
    /// for i in 0..n { use(coeffs[i % 4]) }
    /// ```
    #[inline(always)]
    pub fn get(&self) -> T {
        // SAFETY: read-only from device, and `T: Copy` means we never alias
        // a mutable reference. The host updates this storage only between
        // kernel launches via `cuMemcpyHtoD`, which is synchronized
        // out-of-band relative to device execution.
        unsafe { *self.0.get() }
    }
}
