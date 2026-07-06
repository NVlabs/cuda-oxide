/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graph management (capture, instantiation, and launch).
//!
//! A [`CudaGraph`] represents a directed acyclic graph of GPU operations
//! (kernel launches, memcpys, etc.). Graphs are created either by
//! [stream capture](CudaStream::begin_capture) or by constructing nodes
//! manually via the CUDA graph API. Once built, a graph is
//! [instantiated](CudaGraph::instantiate) into an executable form
//! ([`CudaGraphExec`]) that can be launched on a stream with far less
//! overhead than issuing individual operations.
//!
//! # Stream capture
//!
//! ```ignore
//! let stream = ctx.new_stream()?;
//! stream.begin_capture(CaptureMode::Global)?;
//! // ... enqueue kernels, memcpys, etc. on `stream` ...
//! let graph = stream.end_capture()?;
//! let exec = graph.instantiate()?;
//! exec.launch(&stream)?;
//! ```

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::mem::MaybeUninit;
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Capture mode
// ---------------------------------------------------------------------------

/// Controls how a stream capture sequence interacts with other API calls.
///
/// Mirrors the CUDA driver's `CUstreamCaptureMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    /// Default mode. Unsafe API calls are prohibited when any thread has an
    /// ongoing non-relaxed capture sequence.
    Global,
    /// Only the local thread's capture sequence restricts unsafe API calls.
    ThreadLocal,
    /// The local thread is not restricted from potentially unsafe calls.
    Relaxed,
}

impl CaptureMode {
    fn to_cuda(self) -> cuda_bindings::CUstreamCaptureMode {
        match self {
            Self::Global => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => {
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            }
            Self::Relaxed => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }

    fn from_cuda(mode: cuda_bindings::CUstreamCaptureMode) -> Option<Self> {
        match mode {
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL => {
                Some(Self::Global)
            }
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL => {
                Some(Self::ThreadLocal)
            }
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED => {
                Some(Self::Relaxed)
            }
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// CudaGraph
// ---------------------------------------------------------------------------

/// An RAII wrapper around a `CUgraph` handle.
///
/// Holds an `Arc<CudaContext>` to keep the context alive. Graphs are created
/// via [`CudaGraph::new`] (empty graph for manual construction) or
/// [`CudaStream::end_capture`].
#[derive(Debug, PartialEq, Eq)]
pub struct CudaGraph {
    pub(crate) cu_graph: cuda_bindings::CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUgraph` handles are process-wide; the owning context is kept alive and
/// bound before each driver call.
unsafe impl Send for CudaGraph {}
unsafe impl Sync for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        if !self.cu_graph.is_null() {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { cuda_bindings::cuGraphDestroy(self.cu_graph).result() });
        }
    }
}

impl CudaGraph {
    /// Creates a new empty graph. Nodes and dependencies can be added
    /// manually via the CUDA graph node API.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, DriverError> {
        ctx.bind_to_thread()?;
        let mut cu_graph = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphCreate(cu_graph.as_mut_ptr(), 0).result()?;
            Ok(Self {
                cu_graph: cu_graph.assume_init(),
                ctx: ctx.clone(),
            })
        }
    }

    /// Returns the raw `CUgraph` handle.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }

    /// Instantiates this graph into an executable form.
    ///
    /// The resulting [`CudaGraphExec`] can be launched repeatedly on any stream
    /// with far less overhead than issuing individual operations.
    pub fn instantiate(&self) -> Result<CudaGraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut cu_graph_exec = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(
                cu_graph_exec.as_mut_ptr(),
                self.cu_graph,
                0, // flags: reserved, must be 0
            )
            .result()?;
            Ok(CudaGraphExec {
                cu_graph_exec: cu_graph_exec.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// CudaGraphExec
// ---------------------------------------------------------------------------

/// An executable (instantiated) CUDA graph.
///
/// Created by [`CudaGraph::instantiate`]. Can be launched on any stream
/// via [`launch`](CudaGraphExec::launch) or uploaded to the device for
/// faster repeated launches via [`upload`](CudaGraphExec::upload).
#[derive(Debug, PartialEq, Eq)]
pub struct CudaGraphExec {
    pub(crate) cu_graph_exec: cuda_bindings::CUgraphExec,
    pub(crate) ctx: Arc<CudaContext>,
}

/// # Safety
///
/// `CUgraphExec` handles are process-wide; the owning context is kept alive
/// and bound before each driver call.
unsafe impl Send for CudaGraphExec {}
unsafe impl Sync for CudaGraphExec {}

impl Drop for CudaGraphExec {
    fn drop(&mut self) {
        if !self.cu_graph_exec.is_null() {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx.record_err(unsafe {
                cuda_bindings::cuGraphExecDestroy(self.cu_graph_exec).result()
            });
        }
    }
}

impl CudaGraphExec {
    /// Returns the raw `CUgraphExec` handle.
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_graph_exec
    }

    /// Launches the executable graph on the given stream.
    ///
    /// Only one launch may be in flight at a time per executable graph.
    /// The stream must belong to the same context as the graph.
    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    /// Uploads the executable graph to the device.
    ///
    /// Uploading can reduce launch latency for subsequent launches on
    /// the same device. This is an asynchronous operation; the upload
    /// completes on the device before the next graph launch.
    pub fn upload(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphUpload(self.cu_graph_exec, stream.cu_stream()).result() }
    }
}

// ---------------------------------------------------------------------------
// Stream capture methods (implemented on CudaStream via extension)
// ---------------------------------------------------------------------------

/// Extension methods on [`CudaStream`] for graph capture.
///
/// These are defined as an extension trait rather than inherent methods to
/// keep the stream module focused on core stream operations.
pub trait CudaStreamCaptureExt {
    /// Begins graph capture on this stream.
    ///
    /// All operations enqueued after this call are recorded into a graph
    /// instead of being executed. Capture ends with
    /// [`end_capture`](CudaStreamCaptureExt::end_capture).
    ///
    /// Cannot be called on the default stream (`CU_STREAM_LEGACY`).
    fn begin_capture(&self, mode: CaptureMode) -> Result<(), DriverError>;

    /// Ends graph capture on this stream and returns the captured graph.
    ///
    /// Must be paired with a prior [`begin_capture`](CudaStreamCaptureExt::begin_capture)
    /// on the same stream. If the mode was not [`CaptureMode::Relaxed`],
    /// this must be called from the same thread as `begin_capture`.
    fn end_capture(&self) -> Result<CudaGraph, DriverError>;

    /// Returns whether this stream is currently in an active capture sequence.
    fn is_capturing(&self) -> Result<bool, DriverError>;
}

impl CudaStreamCaptureExt for CudaStream {
    fn begin_capture(&self, mode: CaptureMode) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuStreamBeginCapture_v2(self.cu_stream, mode.to_cuda()).result() }
    }

    fn end_capture(&self) -> Result<CudaGraph, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut cu_graph = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamEndCapture(self.cu_stream, cu_graph.as_mut_ptr()).result()?;
            Ok(CudaGraph {
                cu_graph: cu_graph.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }

    fn is_capturing(&self) -> Result<bool, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut status = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuStreamIsCapturing(self.cu_stream, status.as_mut_ptr()).result()?;
            Ok(status.assume_init() != 0)
        }
    }
}

// ---------------------------------------------------------------------------
// Thread capture mode control
// ---------------------------------------------------------------------------

/// Atomically swaps the calling thread's stream capture interaction mode.
///
/// This is a low-level primitive for controlling how the thread interacts
/// with concurrent capture sequences. Returns the previous mode.
///
/// Prefer [`CaptureModeGuard`] for scoped mode changes that are
/// automatically restored on drop.
///
/// # Example (push-pop pattern)
///
/// ```ignore
/// let old = thread_exchange_capture_mode(CaptureMode::Relaxed)?;
/// // ... do work that should not be restricted by capture ...
/// thread_exchange_capture_mode(old)?;
/// ```
pub fn thread_exchange_capture_mode(mode: CaptureMode) -> Result<CaptureMode, DriverError> {
    let mut raw = mode.to_cuda();
    unsafe {
        cuda_bindings::cuThreadExchangeStreamCaptureMode(&mut raw).result()?;
    }
    CaptureMode::from_cuda(raw)
        .ok_or_else(|| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
}

/// RAII guard that swaps the thread's capture mode on construction and
/// restores the previous mode on drop. Errors during restoration are
/// silently discarded (consistent with destructor semantics).
pub struct CaptureModeGuard {
    prev: CaptureMode,
}

impl CaptureModeGuard {
    /// Swaps the thread's capture mode to `mode` and returns a guard that
    /// will restore the previous mode when dropped.
    pub fn new(mode: CaptureMode) -> Result<Self, DriverError> {
        let prev = thread_exchange_capture_mode(mode)?;
        Ok(Self { prev })
    }
}

impl Drop for CaptureModeGuard {
    fn drop(&mut self) {
        let _ = thread_exchange_capture_mode(self.prev);
    }
}
