/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA context management (primary context, RAII).
//!
//! [`CudaContext`] retains the **primary context** for a given device ordinal
//! via `cuDevicePrimaryCtxRetain` and releases it on [`Drop`]. The primary
//! context is shared across the process; multiple `CudaContext` instances for
//! the same device share the same underlying `CUcontext`.
//!
//! # Thread binding
//!
//! CUDA driver calls are context-scoped and thread-local. [`CudaContext`]
//! transparently calls `cuCtxSetCurrent` before any driver operation, so
//! callers do not need to manage the context stack manually.

use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::ffi::c_int;
use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};

/// Owns a reference to a CUDA device's primary context.
///
/// Created via [`CudaContext::new`] and typically held in an `Arc` so streams,
/// events, and modules can share the same context. Dropping the last reference
/// releases the primary context (`cuDevicePrimaryCtxRelease`).
///
/// Tracks live stream count and accumulated error state atomically for
/// cross-thread diagnostics.
#[derive(Debug)]
pub struct CudaContext {
    /// Raw CUDA device handle (`CUdevice`).
    pub(crate) cu_device: cuda_bindings::CUdevice,
    /// Raw CUDA context handle (`CUcontext`). Set to null on drop.
    pub(crate) cu_ctx: cuda_bindings::CUcontext,
    /// Zero-based device ordinal passed to [`CudaContext::new`].
    pub(crate) ordinal: usize,
    /// Number of live [`CudaStream`] instances sharing this context.
    pub(crate) num_streams: AtomicUsize,
    /// When `true`, the first [`new_stream`](CudaContext::new_stream) call
    /// synchronizes the context to establish a clean ordering baseline.
    pub(crate) event_tracking: AtomicBool,
    /// Sticky error state recorded by [`record_err`](CudaContext::record_err).
    /// Stores the raw `CUresult` value, or `0` if no error.
    pub(crate) error_state: AtomicU32,
}

/// # Safety
///
/// `CUdevice` and `CUcontext` are process-wide handles. All mutable state
/// (`num_streams`, `event_tracking`, `error_state`) uses atomics. The CUDA
/// driver itself is thread-safe for distinct contexts, and the
/// [`bind_to_thread`](CudaContext::bind_to_thread) mechanism ensures the
/// correct context is current before each call.
unsafe impl Send for CudaContext {}
/// See [`Send`] impl.
unsafe impl Sync for CudaContext {}

/// Releases the primary context on drop.
///
/// Binds the context to the current thread first (required by
/// `cuDevicePrimaryCtxRelease`). Errors during teardown are recorded via
/// [`record_err`](CudaContext::record_err) rather than panicking.
impl Drop for CudaContext {
    fn drop(&mut self) {
        self.record_err(self.bind_to_thread());
        let ctx = std::mem::replace(&mut self.cu_ctx, std::ptr::null_mut());
        if !ctx.is_null() {
            self.record_err(unsafe {
                cuda_bindings::cuDevicePrimaryCtxRelease_v2(self.cu_device).result()
            });
        }
    }
}

/// Equality is based on device handle, context handle, and ordinal.
impl PartialEq for CudaContext {
    fn eq(&self, other: &Self) -> bool {
        self.cu_device == other.cu_device
            && self.cu_ctx == other.cu_ctx
            && self.ordinal == other.ordinal
    }
}
impl Eq for CudaContext {}

impl CudaContext {
    /// Creates a new context for the device at `ordinal`.
    ///
    /// Calls [`cuInit`](crate::init), obtains the device handle, retains the
    /// primary context, and binds it to the calling thread. Returns the context
    /// wrapped in an `Arc` for shared ownership across streams, events, and
    /// modules.
    pub fn new(ordinal: usize) -> Result<Arc<Self>, DriverError> {
        unsafe { crate::init(0)? };

        let cu_device = unsafe {
            let mut cu_device = MaybeUninit::uninit();
            cuda_bindings::cuDeviceGet(cu_device.as_mut_ptr(), ordinal as c_int).result()?;
            cu_device.assume_init()
        };

        let cu_ctx = unsafe {
            let mut cu_ctx = MaybeUninit::uninit();
            cuda_bindings::cuDevicePrimaryCtxRetain(cu_ctx.as_mut_ptr(), cu_device).result()?;
            cu_ctx.assume_init()
        };

        let ctx = Arc::new(CudaContext {
            cu_device,
            cu_ctx,
            ordinal,
            num_streams: AtomicUsize::new(0),
            event_tracking: AtomicBool::new(true),
            error_state: AtomicU32::new(0),
        });
        ctx.bind_to_thread()?;
        Ok(ctx)
    }

    /// Returns the zero-based device ordinal.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// Returns the raw `CUdevice` handle.
    pub fn cu_device(&self) -> cuda_bindings::CUdevice {
        self.cu_device
    }

    /// Returns the raw `CUcontext` handle.
    pub fn cu_ctx(&self) -> cuda_bindings::CUcontext {
        self.cu_ctx
    }

    /// Binds this context to the calling thread if not already current.
    ///
    /// Checks [`check_err`](Self::check_err) first and propagates any sticky
    /// error. Skips the `cuCtxSetCurrent` call when the context is already
    /// bound, avoiding an unnecessary driver round-trip.
    ///
    /// Most methods on [`CudaStream`], [`CudaEvent`](crate::CudaEvent), and
    /// [`CudaModule`](crate::CudaModule) call this internally.
    ///
    /// CUcontext is the backing runtime object, and CUmodule / CUfunction / CUstream
    /// are opaque handles to objects created under that context. bind_to_thread()
    /// makes that context, the one, the current host thread is operating against.
    /// If the thread is currently bound to some other context, using those handles
    /// can fail.
    pub fn bind_to_thread(&self) -> Result<(), DriverError> {
        self.check_err()?;
        let mut current = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuCtxGetCurrent(current.as_mut_ptr()).result()?;
            let current = current.assume_init();
            if current.is_null() || current != self.cu_ctx {
                cuda_bindings::cuCtxSetCurrent(self.cu_ctx).result()?;
            }
        }
        Ok(())
    }

    /// Blocks the calling thread until all preceding work in this context
    /// completes.
    ///
    /// Binds the context first, then calls `cuCtxSynchronize`.
    pub fn synchronize(&self) -> Result<(), DriverError> {
        self.bind_to_thread()?;
        unsafe { cuda_bindings::cuCtxSynchronize() }.result()
    }

    /// Returns a handle to the per-context default stream (stream `0`).
    ///
    /// The default stream implicitly synchronizes with all blocking streams in
    /// the same context. The returned [`CudaStream`] holds a null `CUstream`
    /// pointer, which the driver interprets as the default stream.
    pub fn default_stream(self: &Arc<Self>) -> Arc<CudaStream> {
        Arc::new(CudaStream {
            cu_stream: std::ptr::null_mut(),
            ctx: self.clone(),
        })
    }

    /// Creates a new non-blocking stream in this context.
    ///
    /// The stream is created with `CU_STREAM_NON_BLOCKING`, so it does not
    /// implicitly synchronize with the default stream.
    ///
    /// On the first call (when `num_streams` transitions from 0 to 1), the
    /// context is synchronized to establish a clean ordering baseline if
    /// `event_tracking` is enabled.
    pub fn new_stream(self: &Arc<Self>) -> Result<Arc<CudaStream>, DriverError> {
        self.bind_to_thread()?;
        let prev = self.num_streams.fetch_add(1, Ordering::Relaxed);
        if prev == 0 && self.event_tracking.load(Ordering::Relaxed) {
            self.synchronize()?;
        }
        let mut cu_stream = MaybeUninit::uninit();
        let cu_stream = unsafe {
            cuda_bindings::cuStreamCreate(
                cu_stream.as_mut_ptr(),
                cuda_bindings::CUstream_flags_enum_CU_STREAM_NON_BLOCKING,
            )
            .result()?;
            cu_stream.assume_init()
        };
        Ok(Arc::new(CudaStream {
            cu_stream,
            ctx: self.clone(),
        }))
    }

    /// Queries the device's marketing name (e.g. `"NVIDIA GeForce RTX 5090"`).
    ///
    /// Wraps `cuDeviceGetName` with a 256-byte buffer (driver guarantees
    /// the name fits in 256 bytes including the trailing NUL). Returns the
    /// decoded UTF-8 string with any trailing NULs stripped.
    pub fn device_name(&self) -> Result<String, DriverError> {
        self.bind_to_thread()?;
        let mut buf = [0; 256];
        unsafe {
            cuda_bindings::cuDeviceGetName(buf.as_mut_ptr(), buf.len() as c_int, self.cu_device)
                .result()?;
        }
        let bytes: Vec<u8> = buf
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8)
            .collect();
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Queries the compute capability (SM version) of the device.
    ///
    /// Returns `(major, minor)` -- e.g. `(9, 0)` for Hopper H100.
    pub fn compute_capability(&self) -> Result<(i32, i32), DriverError> {
        self.bind_to_thread()?;
        let mut major = MaybeUninit::uninit();
        let mut minor = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuDeviceGetAttribute(
                major.as_mut_ptr(),
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                self.cu_device,
            )
            .result()?;
            cuda_bindings::cuDeviceGetAttribute(
                minor.as_mut_ptr(),
                cuda_bindings::CUdevice_attribute_enum_CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                self.cu_device,
            )
            .result()?;
            Ok((major.assume_init(), minor.assume_init()))
        }
    }

    /// Atomically reads and clears the sticky error state.
    ///
    /// Returns `Ok(())` if no error was recorded, or the stored
    /// [`DriverError`] otherwise. The error is cleared after this call.
    pub fn check_err(&self) -> Result<(), DriverError> {
        let error_state = self.error_state.swap(0, Ordering::Relaxed);
        if error_state == 0 {
            Ok(())
        } else {
            Err(DriverError(error_state))
        }
    }

    /// Records a driver error into the sticky error state.
    ///
    /// Used during [`Drop`] paths where returning a `Result` is not possible.
    /// If `result` is `Err`, the raw error code is stored; subsequent
    /// [`check_err`](Self::check_err) or [`bind_to_thread`](Self::bind_to_thread)
    /// calls will surface it. A later store overwrites an earlier one.
    pub fn record_err<T>(&self, result: Result<T, DriverError>) {
        if let Err(err) = result {
            self.error_state.store(err.0, Ordering::Relaxed)
        }
    }
}
