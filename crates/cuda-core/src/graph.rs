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
//! # What the types enforce
//!
//! **Captured buffers cannot be dropped while the graph can replay.** Capture
//! records raw device *addresses*, not Rust borrows, so the driver can never
//! report which allocations a graph references. [`OwnedGraphExec`] sidesteps that
//! entirely: the state moves into the exec, so nothing outside can name the
//! buffers, let alone free them. The use-after-free is not rejected -- it is
//! unrepresentable.
//!
//! That was worth doing rather than documenting, because the violation is
//! **silent**. Measured on an RTX A2000, replaying a graph whose captured buffer
//! had been freed returned `Ok(())` from launch, synchronize *and* read-back,
//! with the expected data -- the replay read freed memory that still held the old
//! bytes. Stream-ordered deallocation returning pages to a pool makes that the
//! likely presentation, so the failure mode is "passes in testing, corrupts once
//! pages are reused" rather than a fault.
//!
//! **Capture is terminated even when recording fails.** The driver documents that
//! after an error the stream sits in `CU_STREAM_CAPTURE_STATUS_INVALIDATED` and
//!
//! > The capture sequence must be terminated with `cuStreamEndCapture` on the
//! > stream where it was initiated in order to continue using `hStream`.
//!
//! and separately that while a blocking stream is capturing, the legacy null
//! stream is unusable too. So a `record` closure returning `Err` would otherwise
//! strand the stream permanently. An internal RAII guard terminates on that path.
//!
//! **Replays of one exec are serialised.** [`OwnedGraphExec::launch`] takes
//! `&mut self`; two concurrent replays of one `CUgraphExec` are not permitted.
//!
//! # What remains on the caller
//!
//! `launch` *enqueues* a replay; it does not wait. A write issued after it
//! returns is therefore only safe if ordered behind the replay, which holds when
//! both go on the same stream -- `DeviceBuffer`'s copy helpers are all
//! stream-ordered, so the natural usage is correct. Writing from a *different*
//! stream, or from the host without synchronising, races the in-flight graph.
//! That is not expressible here and is documented rather than claimed away.
//!
//! Not modelled, deliberately: node mutation, which interacts with launch
//! contracts (`requires` clauses run on the host at launch, so replay never
//! executes them -- sound only while replay reuses the recorded arguments, which
//! is exactly what mutation would break); graphs containing memory-allocation
//! nodes, where the driver permits at most one live exec per graph; and
//! device-side launch.

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
struct StreamCapture<'stream> {
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
    fn end(mut self) -> Result<CapturedGraph, DriverError> {
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
struct CapturedGraph {
    cu_graph: cuda_bindings::CUgraph,
    ctx: Arc<CudaContext>,
}

impl CapturedGraph {
    /// Number of nodes recorded, useful for confirming a capture saw the work
    /// it was meant to see.
    fn node_count(&self) -> Result<usize, DriverError> {
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
    unsafe fn instantiate(&self) -> Result<GraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut exec = MaybeUninit::<cuda_bindings::CUgraphExec>::uninit();
        let code = unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(exec.as_mut_ptr(), self.cu_graph, 0)
        };
        Ok(GraphExec {
            cu_exec: (code, exec).result()?,
        })
    }

    /// Raw handle, for the parts of the graph surface this module does not wrap.
    #[allow(dead_code)]
    fn cu_graph(&self) -> cuda_bindings::CUgraph {
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
struct GraphExec {
    cu_exec: cuda_bindings::CUgraphExec,
}

impl GraphExec {
    /// Replays the graph on `stream`.
    ///
    /// Takes `&mut self` because concurrent replays of one executable graph are
    /// not permitted; the borrow checker enforces the serialisation that the
    /// driver only documents.
    fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
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
    fn begin_capture(&self, mode: CaptureMode) -> Result<StreamCapture<'_>, DriverError> {
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

/// An executable graph that **owns** the state its capture referenced, and lends
/// it back between replays.
///
/// This is the design the two simpler options fail to reach, and the failures are
/// worth recording because they are what motivates the shape:
///
/// * A lifetime-only exec (carrying `PhantomData<&'captures ()>` from the
///   compile time with no runtime cost — but it holds a shared borrow of every
///   captured buffer, so writing new input between replays is
///   `E0502: cannot borrow as mutable because it is also borrowed as immutable`.
///   Sound, and unusable for the thing graphs are for.
/// * `unsafe instantiate` with a documented obligation is usable and unsound: the
///   violation is silent (`Ok(())` from launch, sync and read, with plausible
///   data).
///
/// Moving the state in resolves both: nothing outside can drop it, and
/// [`state_mut`](Self::state_mut) hands it back for the write-then-replay loop.
///
/// # Stream ordering is the residual obligation
///
/// `launch` *enqueues* a replay; it does not wait for it. So a write issued after
/// `launch` returns is only safe if it is ordered behind the replay, which holds
/// when both go on the same stream — `DeviceBuffer`'s copy helpers are all
/// stream-ordered, so the natural usage is correct. Writing from a *different*
/// stream, or from the host without synchronising, races the in-flight graph.
/// That obligation is not expressible here and is documented rather than claimed.
#[derive(Debug)]
pub struct OwnedGraphExec<S> {
    cu_exec: cuda_bindings::CUgraphExec,
    ctx: Arc<CudaContext>,
    state: S,
    /// Recorded at capture time: the `CUgraph` is released once the executable
    /// graph exists, so the count cannot be queried later.
    nodes: usize,
}

impl<S> OwnedGraphExec<S> {
    /// Replays the graph on `stream`.
    pub fn launch(&mut self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_exec, stream.cu_stream()).result() }
    }

    /// The captured state, for writing new input before the next replay.
    ///
    /// `&mut self` means this cannot overlap a `launch` call, though see the note
    /// on stream ordering: it does not order against a replay already in flight.
    pub fn state_mut(&mut self) -> &mut S {
        &mut self.state
    }

    /// The captured state, immutably.
    pub fn state(&self) -> &S {
        &self.state
    }

    /// Nodes recorded by the capture.
    ///
    /// Useful for confirming a capture saw the work it was meant to: a capture
    /// that silently recorded nothing still replays successfully, and does
    /// nothing.
    pub fn node_count(&self) -> usize {
        self.nodes
    }

    /// Gives the captured state back, destroying the graph.
    pub fn into_state(self) -> S {
        // Move `state` out without running `Drop for OwnedGraphExec`, then free
        // the exec by hand so the graph is not leaked.
        let me = core::mem::ManuallyDrop::new(self);
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(me.cu_exec);
            core::ptr::read(&me.state)
        }
    }
}

impl<S> Drop for OwnedGraphExec<S> {
    fn drop(&mut self) {
        unsafe {
            let _ = cuda_bindings::cuGraphExecDestroy(self.cu_exec);
        }
    }
}

impl CudaStream {
    /// Captures work that operates on `state`, and returns an executable graph
    /// owning it.
    ///
    /// `record` receives `&mut S` and must enqueue its work on this stream. The
    /// state is then reachable through [`OwnedGraphExec::state_mut`] for the
    /// write-then-replay loop, and recoverable with
    /// [`OwnedGraphExec::into_state`].
    ///
    /// Safe, because nothing outside the exec can drop the buffers the capture
    /// referenced. See the type docs for the stream-ordering obligation that
    /// remains.
    pub fn capture_owning<S, F>(
        &self,
        mode: CaptureMode,
        mut state: S,
        record: F,
    ) -> Result<OwnedGraphExec<S>, DriverError>
    where
        F: FnOnce(&mut S) -> Result<(), DriverError>,
    {
        let capture = self.begin_capture(mode)?;
        // On `Err`, dropping the guard terminates the capture, so the stream
        // stays usable and `state` is returned to the caller by being dropped.
        record(&mut state)?;
        let graph = capture.end()?;
        let nodes = graph.node_count()?;
        // SAFETY: `state` is moved into the returned exec, so every buffer the
        // capture referenced through it outlives every replay by construction --
        // nothing outside the exec can name them, let alone drop them.
        let exec = unsafe { graph.instantiate() }?;
        let cu_exec = exec.cu_graph_exec();
        // Transfer the handle rather than letting both wrappers free it.
        core::mem::forget(exec);
        Ok(OwnedGraphExec {
            cu_exec,
            ctx: Arc::clone(self.context()),
            state,
            nodes,
        })
    }
}
