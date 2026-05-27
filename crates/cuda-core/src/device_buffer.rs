/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Owning device memory buffer with ergonomic host-device transfer methods.
//!
//! [`DeviceBuffer<T>`] is analogous to `Vec<T>` on the host: it owns a
//! contiguous allocation of `len` elements on the device and frees it on
//! drop. Unlike cudarc's `CudaSlice`, the buffer carries no stream reference
//! and no hidden event tracking -- the stream is an explicit parameter on
//! every transfer operation, making data-flow and synchronization transparent.
//!
//! # Quick start
//!
//! ```ignore
//! let a_dev = DeviceBuffer::from_host(&stream, &a_host)?;
//! let c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;
//! // ... kernel launch ...
//! let c_host = c_dev.to_host_vec(&stream)?;
//! ```

use std::marker::PhantomData;
use std::sync::Arc;

use cuda_bindings::CUdeviceptr;

use crate::context::CudaContext;
use crate::error::DriverError;
use crate::pinned_host_buffer::PinnedHostBuffer;
use crate::stream::CudaStream;

/// Marker trait for values that can be safely copied between host and device
/// memory as raw bytes.
///
/// Types implementing `DeviceCopy` must not contain Rust-owned allocations,
/// references, or other values whose validity depends on host-side ownership or
/// drop semantics. This is the device-memory equivalent of a plain-old-data
/// contract.
///
/// # Safety
///
/// Implementors must be safe to duplicate with a byte-for-byte copy. Values
/// copied back from device memory must have a bit pattern that is valid for
/// `Self`, and the all-zero bit pattern must also be valid because
/// [`DeviceBuffer::zeroed`] initializes memory with zero bytes.
///
/// `Copy` alone is not enough: types such as `bool`, `char`, and
/// `NonZeroU32` are `Copy`, but not every byte pattern is a valid value of
/// those types. `DeviceCopy` is the stronger promise required when
/// `DeviceBuffer` turns raw device bytes back into initialized Rust values.
pub unsafe trait DeviceCopy: Copy {}

macro_rules! impl_device_copy {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl DeviceCopy for $ty {}
        )+
    };
}

impl_device_copy!(
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

unsafe impl<T: DeviceCopy, const N: usize> DeviceCopy for [T; N] {}
unsafe impl<T: ?Sized> DeviceCopy for *const T {}
unsafe impl<T: ?Sized> DeviceCopy for *mut T {}

macro_rules! impl_device_copy_tuple {
    ($($name:ident),+ $(,)?) => {
        unsafe impl<$($name: DeviceCopy),+> DeviceCopy for ($($name,)+) {}
    };
}

impl_device_copy_tuple!(A);
impl_device_copy_tuple!(A, B);
impl_device_copy_tuple!(A, B, C);
impl_device_copy_tuple!(A, B, C, D);
impl_device_copy_tuple!(A, B, C, D, E);
impl_device_copy_tuple!(A, B, C, D, E, F);
impl_device_copy_tuple!(A, B, C, D, E, F, G);
impl_device_copy_tuple!(A, B, C, D, E, F, G, H);

unsafe impl DeviceCopy for half::bf16 {}
unsafe impl DeviceCopy for half::f16 {}

/// Owning handle to a contiguous device allocation of `T` elements.
///
/// Holds a raw device pointer, element count, and a reference-counted
/// context that keeps the CUDA context alive. Dropping the buffer calls
/// `cuMemFree` (synchronous); for async-sensitive workloads, use
/// `cuda_async::DeviceBox` which frees via a deallocator stream.
///
/// Device buffers may only transfer plain device-copyable values. Owning host
/// types such as [`String`] are rejected because copying their bytes to and
/// from device memory would not preserve Rust ownership invariants.
///
/// ```compile_fail
/// # use cuda_core::{CudaStream, DeviceBuffer};
/// # fn rejects_non_device_copy(stream: &CudaStream) {
/// let _ = DeviceBuffer::<String>::zeroed(stream, 1);
/// # }
/// ```
pub struct DeviceBuffer<T> {
    ptr: CUdeviceptr,
    len: usize,
    ctx: Arc<CudaContext>,
    _marker: PhantomData<T>,
}

// SAFETY: CUdeviceptr is a u64 handle valid across threads when the owning
// context is bound. The PhantomData<T> is Send if T is Send.
unsafe impl<T: Send> Send for DeviceBuffer<T> {}
// SAFETY: &DeviceBuffer only exposes cu_deviceptr() and len(), both of which
// return Copy values. No interior mutability.
unsafe impl<T: Send + Sync> Sync for DeviceBuffer<T> {}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.ptr != 0 {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { crate::memory::free_sync(self.ptr) });
        }
    }
}

impl<T> DeviceBuffer<T> {
    /// Returns the raw `CUdeviceptr` for use in kernel argument lists.
    #[inline]
    pub fn cu_deviceptr(&self) -> CUdeviceptr {
        self.ptr
    }

    /// Number of `T` elements in the buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer has zero elements.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Total size in bytes (`len * size_of::<T>()`).
    #[inline]
    pub fn num_bytes(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    /// Returns a reference to the owning context.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Constructs a `DeviceBuffer` from pre-existing raw parts.
    ///
    /// # Safety
    ///
    /// - `ptr` must have been allocated via `cuMemAlloc*` with at least
    ///   `len * size_of::<T>()` bytes.
    /// - `ptr` must belong to the same CUDA context as `ctx`.
    /// - The caller transfers ownership -- `ptr` will be freed on drop.
    pub unsafe fn from_raw_parts(ptr: CUdeviceptr, len: usize, ctx: Arc<CudaContext>) -> Self {
        Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        }
    }

    /// Consumes the buffer and returns the raw parts without freeing.
    ///
    /// The caller is responsible for eventually freeing `ptr`.
    pub fn into_raw_parts(self) -> (CUdeviceptr, usize, Arc<CudaContext>) {
        let parts = (self.ptr, self.len, self.ctx.clone());
        std::mem::forget(self);
        parts
    }
}

impl<T: DeviceCopy> DeviceBuffer<T> {
    /// Allocates device memory and copies `data` from the host, enqueued on
    /// `stream`.
    ///
    /// The host slice must remain valid until the copy completes (i.e. until
    /// the next synchronization point on `stream`). For pageable host memory
    /// the driver may internally synchronize; use pinned memory for true
    /// async overlap.
    pub fn from_host(stream: &CudaStream, data: &[T]) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let len = data.len();
        let num_bytes = std::mem::size_of_val(data);

        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        unsafe {
            crate::memory::memcpy_htod_async(ptr, data.as_ptr(), num_bytes, stream.cu_stream())?;
        }
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Allocates device memory and enqueues a host-to-device copy from a
    /// pinned host buffer on `stream`, returning without synchronizing.
    ///
    /// Pinned host memory allows CUDA to avoid the pageable-memory staging
    /// path and is required when host-device copies need true asynchronous
    /// overlap with other stream work.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `data` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// The device-to-host counterparts are [`Self::copy_to_pinned_host`]
    /// (blocking) and [`Self::copy_to_pinned_host_async`] (non-blocking). To
    /// refill an existing device buffer instead of allocating a new one, use
    /// [`Self::copy_from_pinned_host_async`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy on `stream` and
    /// returns; CUDA may still be reading from `data`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `data` is not dropped, freed, mutated, or aliased until the enqueued
    /// copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `data` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn from_pinned_host(
        stream: &CudaStream,
        data: &PinnedHostBuffer<T>,
    ) -> Result<Self, DriverError> {
        debug_assert!(
            Arc::ptr_eq(data.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        Self::from_host(stream, data.as_slice())
    }

    /// Allocates zero-initialized device memory of `len` elements, enqueued
    /// on `stream`.
    pub fn zeroed(stream: &CudaStream, len: usize) -> Result<Self, DriverError> {
        let ctx = stream.context().clone();
        let num_bytes = len * std::mem::size_of::<T>();

        let ptr = unsafe { crate::memory::malloc_async(stream.cu_stream(), num_bytes)? };
        if num_bytes > 0 {
            unsafe {
                crate::memory::memset_d8_async(ptr, 0, num_bytes, stream.cu_stream())?;
            }
        }
        Ok(Self {
            ptr,
            len,
            ctx,
            _marker: PhantomData,
        })
    }

    /// Copies the entire buffer back to the host, returning a `Vec<T>`.
    ///
    /// Synchronizes on `stream` before returning so the host vector is safe
    /// to read immediately.
    pub fn to_host_vec(&self, stream: &CudaStream) -> Result<Vec<T>, DriverError> {
        let mut host = Vec::with_capacity(self.len);
        unsafe {
            crate::memory::memcpy_dtoh_async(
                host.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()?;
        unsafe { host.set_len(self.len) };
        Ok(host)
    }

    /// Copies the buffer contents into an existing host slice.
    ///
    /// Synchronizes on `stream` before returning. Panics if
    /// `dst.len() < self.len()`.
    pub fn copy_to_host(&self, stream: &CudaStream, dst: &mut [T]) -> Result<(), DriverError> {
        assert!(
            dst.len() >= self.len,
            "destination slice too small: {} < {}",
            dst.len(),
            self.len
        );
        unsafe {
            crate::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )?;
        }
        stream.synchronize()
    }

    /// Copies the buffer contents into an existing pinned host buffer and
    /// synchronizes `stream` before returning.
    ///
    /// Panics if `dst.len() < self.len()`. Use pinned destinations when you
    /// need the transfer to avoid pageable-memory staging; this helper still
    /// waits for completion before returning, matching [`Self::copy_to_host`].
    ///
    /// For true DtoH overlap, use [`Self::copy_to_pinned_host_async`] and
    /// synchronize the stream later.
    pub fn copy_to_pinned_host(
        &self,
        stream: &CudaStream,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        // SAFETY: we synchronize the stream below before returning, so the
        // pinned destination is no longer being written to by CUDA when the
        // mutable borrow on `dst` is released to the caller.
        unsafe { self.copy_to_pinned_host_async(stream, dst)? };
        stream.synchronize()
    }

    /// Enqueues a device-to-host copy into an existing pinned host buffer and
    /// returns without synchronizing.
    ///
    /// Panics if `dst.len() < self.len()`.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `dst` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the device-to-host copy on `stream` and
    /// returns; CUDA may still be writing into `dst`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `dst` is not dropped, freed, read, or aliased until the enqueued copy
    /// has completed, typically after the next [`CudaStream::synchronize`]
    /// call or a stream-ordered event wait. Dropping `dst` before that
    /// synchronization point calls `cuMemFreeHost` while the in-flight
    /// transfer is still writing the buffer, which is undefined behavior.
    pub unsafe fn copy_to_pinned_host_async(
        &self,
        stream: &CudaStream,
        dst: &mut PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(dst.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        assert!(
            dst.len() >= self.len,
            "destination pinned host buffer too small: {} < {}",
            dst.len(),
            self.len
        );
        unsafe {
            crate::memory::memcpy_dtoh_async(
                dst.as_mut_ptr(),
                self.ptr,
                self.num_bytes(),
                stream.cu_stream(),
            )
        }
    }

    /// Enqueues a host-to-device copy from a pinned host buffer into this
    /// device buffer and returns without synchronizing.
    ///
    /// This is the symmetric counterpart of
    /// [`Self::copy_to_pinned_host_async`]: it refills an existing device
    /// allocation from rotating pinned host stagers instead of allocating a
    /// fresh device buffer per refresh, which is the typical shape for
    /// asynchronous overlap pipelines.
    ///
    /// Panics if `src.len() > self.len()`.
    ///
    /// `PinnedHostBuffer` currently uses `cuMemAllocHost` without the
    /// `PORTABLE` flag, so the allocation is only pinned in the context that
    /// created it. In debug builds this asserts that `src` and `stream`
    /// share the same [`CudaContext`].
    ///
    /// # Safety
    ///
    /// This call only enqueues the host-to-device copy on `stream` and
    /// returns; CUDA may still be reading from `src`'s pinned pointer long
    /// after this function returns. The caller is responsible for ensuring
    /// `src` is not dropped, freed, mutated, or aliased until the enqueued
    /// copy has completed, typically after the next
    /// [`CudaStream::synchronize`] call or a stream-ordered event wait.
    /// Dropping `src` before that synchronization point calls
    /// `cuMemFreeHost` while the in-flight transfer is still reading the
    /// buffer, which is undefined behavior.
    pub unsafe fn copy_from_pinned_host_async(
        &mut self,
        stream: &CudaStream,
        src: &PinnedHostBuffer<T>,
    ) -> Result<(), DriverError> {
        debug_assert!(
            Arc::ptr_eq(src.context(), stream.context()),
            "pinned host buffer and stream must belong to the same CUDA context"
        );
        assert!(
            src.len() <= self.len,
            "source pinned host buffer too large: {} > {}",
            src.len(),
            self.len
        );
        let num_bytes = src.num_bytes();
        unsafe {
            crate::memory::memcpy_htod_async(self.ptr, src.as_ptr(), num_bytes, stream.cu_stream())
        }
    }
}
