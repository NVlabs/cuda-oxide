/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA graph capture and replay (RAII), **exploratory**.
//!
//! This module exists to find out what a safe graph API can and cannot promise.
//! It wraps the smallest useful slice of the graph surface — stream capture,
//! instantiation, replay — and records, in prose next to each item, which
//! obligations the type system discharges and which it merely documents.
//!
//! # Why bother
//!
//! Graph replay is worth a fixed ~1 µs per launch. Measured on this repo's
//! benchmark harness (RTX A2000, sm_86, 500 iterations):
//!
//! ```text
//! path    direct P50   graph P50   saved    nodes   µs/node
//! FP16    3.867 ms     3.697 ms    0.170    173     0.98
//! INT8    4.670 ms     4.491 ms    0.179    222     0.81
//! W8A16   9.180 ms     9.002 ms    0.178    293     0.61
//! ```
//!
//! The saving is near-constant while node counts differ by 70%, which is what
//! identifies it as per-launch overhead rather than anything kernel-related.
//!
//! # What the types actually enforce
//!
//! **Capture must be terminated even when it fails.** This is the obligation
//! most easily missed and the main reason to prefer a guard. The driver
//! documents that after an error, the stream enters
//! `CU_STREAM_CAPTURE_STATUS_INVALIDATED` and
//!
//! > The capture sequence must be terminated with `cuStreamEndCapture` on the
//! > stream where it was initiated in order to continue using `hStream`.
//!
//! So an early return or panic between begin and end leaves the stream
//! permanently unusable — and, because a capturing blocking stream also puts
//! the legacy null stream in "an unusable state", it can take the default
//! stream down with it. [`StreamCapture`] terminates in [`Drop`], so unwinding
//! cannot poison the stream.
//!
//! **Replays of one executable graph are serialised.** [`GraphExec::launch`]
//! takes `&mut self`. Two concurrent replays of the same `CUgraphExec` are not
//! permitted, and `&mut` is exactly that constraint, checked at compile time
//! rather than documented.
//!
//! # What the types do NOT enforce — read before relying on this
//!
//! **Captured graphs freeze device pointers, and nothing here ties their
//! lifetimes together.** Capture records the *argument values* that were
//! enqueued, including raw device addresses. If a buffer used during capture is
//! dropped while a [`GraphExec`] built from that capture is still alive, replay
//! dereferences freed device memory. There is no borrow relating the two:
//!
//! ```ignore
//! let exec = {
//!     let scratch = DeviceBuffer::<f32>::new(&ctx, n)?;   // dropped here
//!     let capture = stream.begin_capture(CaptureMode::Global)?;
//!     launch_using(&scratch)?;
//!     capture.end()?.instantiate()?
//! };
//! exec.launch(&stream)?;   // use-after-free; compiles today
//! ```
//!
//! **Measured, not assumed** (`tests/graph_capture.rs`, the `--ignored` probe, on
//! an RTX A2000). The driver does not diagnose this, and the result looks right:
//!
//! ```text
//! launch    = Ok(())
//! sync      = Ok(())
//! read      = Ok(())
//! dst       = [7, 8, 9, 10]   (captured source was [7, 8, 9, 10])
//! diagnosed = false
//! ```
//!
//! The replay read freed device memory that still happened to hold the old
//! bytes, so the copy "succeeded". That is the worst available failure mode: it
//! passes in testing and corrupts once the freed pages are reused, with no error
//! at launch, at synchronize, or on read-back. Stream-ordered deallocation
//! returning memory to a pool rather than the driver makes this the *likely*
//! presentation, not a rare one.
//!
//! Expressing the constraint would require the exec to borrow every buffer
//! touched during capture, which the capture API cannot observe — the driver
//! sees pointer values, not Rust borrows. Options, none free: thread a lifetime
//! through a capture-scope closure that owns the buffers, or make graphs own
//! `Arc` clones of their inputs.
//!
//! **[`CapturedGraph::instantiate`] is therefore `unsafe`**, carrying this as its
//! documented obligation. That is the honest signature given the failure is
//! silent: a safe one would promise a check nothing performs. Everything else
//! here is safe, so the `unsafe` marks exactly the one step that takes on the
//! lifetime responsibility.
//!
//! Also not modelled, deliberately: graphs containing memory allocation nodes
//! (the driver permits at most one live exec per graph in that case), device
//! launch, and node mutation. Mutation in particular interacts with launch
//! contracts — see the note on [`GraphExec`].

use crate::context::CudaContext;
use crate::error::{DriverError, IntoResult};
use crate::stream::CudaStream;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// How a capture treats concurrent activity in other threads.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CaptureMode {
    /// Prohibit potentially unsafe driver actions in *any* thread for the
    /// duration of the capture. The conservative default.
    #[default]
    Global,
    /// Prohibit them only in the capturing thread.
    ThreadLocal,
    /// Prohibit nothing; the caller guarantees no interfering action occurs.
    Relaxed,
}

impl CaptureMode {
    fn as_raw(self) -> cuda_bindings::CUstreamCaptureMode {
        match self {
            Self::Global => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => {
                cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
            }
            Self::Relaxed => cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

/// An in-progress stream capture.
///
/// Work enqueued on the stream while this guard is alive is recorded into a
/// graph instead of executing. Call [`end`](Self::end) to finish and obtain the
/// [`CapturedGraph`].
///
/// Dropping the guard without calling `end` terminates the capture and discards
/// the graph. That is not merely tidy: an unterminated capture leaves the stream
/// unusable, so termination has to happen on the unwind path too.
#[must_use = "dropping the guard immediately ends the capture and discards the graph"]
pub struct StreamCapture<'stream> {
    stream: &'stream CudaStream,
    /// Cleared by `end` so `Drop` does not terminate the capture twice.
    active: bool,
}

impl<'stream> StreamCapture<'stream> {
    /// Finishes the capture and returns the recorded graph.
    ///
    /// Returns an error if the capture was invalidated. Note that ending an
    /// invalidated capture is *required* to make the stream usable again, so
    /// the error path here has already done the necessary cleanup.
    pub fn end(mut self) -> Result<CapturedGraph, DriverError> {
        self.active = false;
        self.stream.context().bind_to_thread()?;
        let mut graph = MaybeUninit::<cuda_bindings::CUgraph>::uninit();
        let code = unsafe {
            cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), graph.as_mut_ptr())
        };
        let cu_graph = (code, graph).result()?;
        if cu_graph.is_null() {
            // `cuStreamEndCapture` reports success with a null graph when the
            // capture was invalidated by an earlier error, so a bare status
            // check is not enough to conclude a graph exists.
            return Err(DriverError(
                cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
            ));
        }
        Ok(CapturedGraph {
            cu_graph,
            ctx: Arc::clone(self.stream.context()),
        })
    }
}

impl Drop for StreamCapture<'_> {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Terminate and discard. Errors are unreportable here, but the call
        // must still happen: skipping it leaves the stream unusable, which is a
        // worse outcome than a leaked graph handle.
        let mut graph = MaybeUninit::<cuda_bindings::CUgraph>::uninit();
        unsafe {
            let _ = cuda_bindings::cuStreamEndCapture(self.stream.cu_stream(), graph.as_mut_ptr());
            let handle = graph.assume_init();
            if !handle.is_null() {
                let _ = cuda_bindings::cuGraphDestroy(handle);
            }
        }
    }
}

/// A captured, not-yet-executable graph (`CUgraph`).
#[derive(Debug)]
pub struct CapturedGraph {
    cu_graph: cuda_bindings::CUgraph,
    ctx: Arc<CudaContext>,
}

impl CapturedGraph {
    /// Number of nodes recorded, useful for confirming a capture saw the work
    /// it was meant to see.
    pub fn node_count(&self) -> Result<usize, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut count: usize = 0;
        unsafe {
            cuda_bindings::cuGraphGetNodes(self.cu_graph, std::ptr::null_mut(), &mut count)
                .result()?;
        }
        Ok(count)
    }

    /// Instantiates an executable graph.
    ///
    /// # Safety
    ///
    /// Every device allocation referenced by the captured work must remain live,
    /// and at the same address, for as long as the returned [`GraphExec`] can be
    /// replayed. Capture records raw device addresses, not Rust borrows, so
    /// nothing relates the two lifetimes and the compiler cannot help.
    ///
    /// This is `unsafe` because the violation is **silent**. Measured on an
    /// RTX A2000 (`tests/graph_capture.rs`, the `--ignored` probe), replaying a
    /// graph whose captured buffer had been dropped returned `Ok(())` from launch,
    /// synchronize *and* read-back, with the expected data — the replay read freed
    /// memory that still held the old bytes. Stream-ordered deallocation returning
    /// pages to a pool rather than the driver makes that the likely presentation,
    /// so the failure mode is "passes in testing, corrupts once pages are reused"
    /// rather than a fault. A safe signature would promise a check that neither
    /// the driver nor the type system performs.
    ///
    /// See the module docs for the three ways this could be closed; picking among
    /// them is a design decision, so this function states the obligation instead.
    pub unsafe fn instantiate(&self) -> Result<GraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec = MaybeUninit::<cuda_bindings::CUgraphExec>::uninit();
        let code = unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), self.cu_graph, 0)
        };
        Ok(GraphExec {
            cu_exec: (code, exec).result()?,
            ctx: Arc::clone(&self.ctx),
        })
    }

    /// Raw handle, for the parts of the graph surface this module does not wrap.
    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphDestroy(self.cu_graph);
        }
    }
}

/// An executable graph (`CUgraphExec`), ready for replay.
///
/// # Interaction with `launch_contract`
///
/// Kernels carrying a `requires` launch contract validate their size relations
/// **on the host, at launch**. Replay runs no host code, so those checks do not
/// execute per replay. That is sound as long as replay reuses the arguments
/// recorded at capture — each distinct configuration gets its own capture and
/// therefore its own validation — and the guarantee degrades from "every launch
/// is checked" to "this graph was validated when it was built". Mutating node
/// parameters after capture would break it, which is why this module does not
/// wrap `cuGraphExecKernelNodeSetParams`.
#[derive(Debug)]
pub struct GraphExec {
    cu_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
}

impl GraphExec {
    /// Replays the graph on `stream`.
    ///
    /// Takes `&mut self` because concurrent replays of one executable graph are
    /// not permitted; the borrow checker enforces the serialisation that the
    /// driver only documents.
    pub fn launch(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_exec, stream.cu_stream()).result() }
    }

    /// Raw handle, for the parts of the graph surface this module does not wrap.
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_exec
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(self.cu_exec);
        }
    }
}

impl CudaStream {
    /// Begins capturing work enqueued on this stream into a graph.
    ///
    /// The returned guard must be ended (or dropped) before the stream is used
    /// normally again.
    pub fn begin_capture(&self, mode: CaptureMode) -> Result<StreamCapture<'_>, DriverError> {
        self.context().bind_to_thread()?;
        unsafe {
            cuda_bindings::cuStreamBeginCapture_v2(self.cu_stream(), mode.as_raw()).result()?;
        }
        Ok(StreamCapture {
            stream: self,
            active: true,
        })
    }

    /// Whether this stream is currently capturing, and whether that capture has
    /// been invalidated by an error.
    pub fn capture_status(&self) -> Result<CaptureStatus, DriverError> {
        self.context().bind_to_thread()?;
        let mut status = MaybeUninit::<cuda_bindings::CUstreamCaptureStatus>::uninit();
        let code =
            unsafe { cuda_bindings::cuStreamIsCapturing(self.cu_stream(), status.as_mut_ptr()) };
        Ok(match (code, status).result()? {
            cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_ACTIVE => {
                CaptureStatus::Active
            }
            cuda_bindings::CUstreamCaptureStatus_enum_CU_STREAM_CAPTURE_STATUS_INVALIDATED => {
                CaptureStatus::Invalidated
            }
            _ => CaptureStatus::None,
        })
    }
}

/// Capture state of a stream.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CaptureStatus {
    /// Not capturing.
    None,
    /// Capturing normally.
    Active,
    /// Was capturing; an error invalidated the sequence. The capture must still
    /// be ended before the stream can be used again.
    Invalidated,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_mode_maps_to_driver_constants() {
        assert_eq!(
            CaptureMode::Global.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_GLOBAL
        );
        assert_eq!(
            CaptureMode::ThreadLocal.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        );
        assert_eq!(
            CaptureMode::Relaxed.as_raw(),
            cuda_bindings::CUstreamCaptureMode_enum_CU_STREAM_CAPTURE_MODE_RELAXED
        );
        // Global is the conservative choice, so it must be the default.
        assert_eq!(CaptureMode::default(), CaptureMode::Global);
    }
}
