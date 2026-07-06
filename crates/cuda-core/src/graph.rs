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
//!
//! # Manual construction
//!
//! ```ignore
//! let graph = CudaGraph::new(&ctx)?;
//! let kernel_node = graph.add_kernel_node(&[], &kernel_params)?;
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureMode {
    Global,
    ThreadLocal,
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

#[derive(Debug, PartialEq, Eq)]
pub struct CudaGraph {
    pub(crate) cu_graph: cuda_bindings::CUgraph,
    pub(crate) ctx: Arc<CudaContext>,
}

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

    pub fn cu_graph(&self) -> cuda_bindings::CUgraph {
        self.cu_graph
    }

    pub fn instantiate(&self) -> Result<CudaGraphExec, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut cu_graph_exec = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphInstantiateWithFlags(
                cu_graph_exec.as_mut_ptr(),
                self.cu_graph,
                0,
            )
            .result()?;
            Ok(CudaGraphExec {
                cu_graph_exec: cu_graph_exec.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }

    // -- Node construction --

    pub fn add_kernel_node(
        &self,
        dependencies: &[&CudaGraphNode],
        params: &KernelNodeParams,
    ) -> Result<CudaGraphNode, DriverError> {
        self.ctx.bind_to_thread()?;
        let dep_nodes: Vec<cuda_bindings::CUgraphNode> =
            dependencies.iter().map(|n| n.cu_node).collect();
        let (mut ptrs, _buf) = params.to_raw();

        let cu_params = cuda_bindings::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: ptrs.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };

        let mut cu_node = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphAddKernelNode_v2(
                cu_node.as_mut_ptr(),
                self.cu_graph,
                if dep_nodes.is_empty() {
                    std::ptr::null()
                } else {
                    dep_nodes.as_ptr()
                },
                dep_nodes.len(),
                &cu_params,
            )
            .result()?;
            Ok(CudaGraphNode {
                cu_node: cu_node.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }

    pub fn add_empty_node(
        &self,
        dependencies: &[&CudaGraphNode],
    ) -> Result<CudaGraphNode, DriverError> {
        self.ctx.bind_to_thread()?;
        let dep_nodes: Vec<cuda_bindings::CUgraphNode> =
            dependencies.iter().map(|n| n.cu_node).collect();

        let mut cu_node = MaybeUninit::uninit();
        unsafe {
            cuda_bindings::cuGraphAddEmptyNode(
                cu_node.as_mut_ptr(),
                self.cu_graph,
                if dep_nodes.is_empty() {
                    std::ptr::null()
                } else {
                    dep_nodes.as_ptr()
                },
                dep_nodes.len(),
            )
            .result()?;
            Ok(CudaGraphNode {
                cu_node: cu_node.assume_init(),
                ctx: self.ctx.clone(),
            })
        }
    }

    pub fn add_dependencies(
        &self,
        from: &[&CudaGraphNode],
        to: &[&CudaGraphNode],
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        if from.is_empty() || to.is_empty() {
            return Ok(());
        }
        let from_nodes: Vec<cuda_bindings::CUgraphNode> = from.iter().map(|n| n.cu_node).collect();
        let to_nodes: Vec<cuda_bindings::CUgraphNode> = to.iter().map(|n| n.cu_node).collect();
        let count = from.len().min(to.len());
        unsafe {
            cuda_bindings::cuGraphAddDependencies_v2(
                self.cu_graph,
                from_nodes.as_ptr(),
                to_nodes.as_ptr(),
                std::ptr::null(),
                count,
            )
            .result()
        }
    }
}

// ---------------------------------------------------------------------------
// CudaGraphExec
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub struct CudaGraphExec {
    pub(crate) cu_graph_exec: cuda_bindings::CUgraphExec,
    pub(crate) ctx: Arc<CudaContext>,
}

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
    pub fn cu_graph_exec(&self) -> cuda_bindings::CUgraphExec {
        self.cu_graph_exec
    }

    pub fn launch(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphLaunch(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    pub fn upload(&self, stream: &CudaStream) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        unsafe { cuda_bindings::cuGraphUpload(self.cu_graph_exec, stream.cu_stream()).result() }
    }

    // -- Update --

    pub fn update(&self, updated_graph: &CudaGraph) -> Result<GraphUpdateResult, DriverError> {
        self.ctx.bind_to_thread()?;
        let mut result_info = cuda_bindings::CUgraphExecUpdateResultInfo {
            result: cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_SUCCESS,
            errorNode: std::ptr::null_mut(),
            errorFromNode: std::ptr::null_mut(),
        };
        let cu_result = unsafe {
            cuda_bindings::cuGraphExecUpdate_v2(
                self.cu_graph_exec,
                updated_graph.cu_graph,
                &mut result_info,
            )
        };
        if cu_result == 0 {
            return Ok(GraphUpdateResult::Success);
        }
        Ok(match result_info.result {
            cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED => GraphUpdateResult::TopologyChanged,
            cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED => GraphUpdateResult::NodeTypeChanged,
            cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED => GraphUpdateResult::FunctionChanged,
            cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED => GraphUpdateResult::ParameterChanged,
            cuda_bindings::CUgraphExecUpdateResult_enum_CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED => GraphUpdateResult::NotSupported,
            _ => GraphUpdateResult::Unspecified,
        })
    }

    pub fn set_kernel_node_params(
        &self,
        node: &CudaGraphNode,
        params: &KernelNodeParams,
    ) -> Result<(), DriverError> {
        self.ctx.bind_to_thread()?;
        let (mut ptrs, _buf) = params.to_raw();
        let cu_params = cuda_bindings::CUDA_KERNEL_NODE_PARAMS {
            func: params.func,
            gridDimX: params.grid_dim.0,
            gridDimY: params.grid_dim.1,
            gridDimZ: params.grid_dim.2,
            blockDimX: params.block_dim.0,
            blockDimY: params.block_dim.1,
            blockDimZ: params.block_dim.2,
            sharedMemBytes: params.shared_mem_bytes,
            kernelParams: ptrs.as_mut_ptr(),
            extra: std::ptr::null_mut(),
            kern: std::ptr::null_mut(),
            ctx: std::ptr::null_mut(),
        };
        unsafe {
            cuda_bindings::cuGraphExecKernelNodeSetParams_v2(
                self.cu_graph_exec,
                node.cu_node,
                &cu_params,
            )
            .result()
        }
    }
}

// ---------------------------------------------------------------------------
// CudaGraphNode
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
pub struct CudaGraphNode {
    pub(crate) cu_node: cuda_bindings::CUgraphNode,
    pub(crate) ctx: Arc<CudaContext>,
}

unsafe impl Send for CudaGraphNode {}
unsafe impl Sync for CudaGraphNode {}

impl Drop for CudaGraphNode {
    fn drop(&mut self) {
        if !self.cu_node.is_null() {
            self.ctx.record_err(self.ctx.bind_to_thread());
            self.ctx
                .record_err(unsafe { cuda_bindings::cuGraphDestroyNode(self.cu_node).result() });
        }
    }
}

impl CudaGraphNode {
    pub fn cu_node(&self) -> cuda_bindings::CUgraphNode {
        self.cu_node
    }
}

// ---------------------------------------------------------------------------
// KernelNodeParams
// ---------------------------------------------------------------------------

/// Parameters for a kernel launch node in a CUDA graph.
pub struct KernelNodeParams {
    pub func: cuda_bindings::CUfunction,
    pub grid_dim: (u32, u32, u32),
    pub block_dim: (u32, u32, u32),
    pub shared_mem_bytes: u32,
    kernel_params_bytes: Vec<u8>,
    kernel_param_offsets: Vec<usize>,
}

impl KernelNodeParams {
    pub fn new(
        func: cuda_bindings::CUfunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        shared_mem: u32,
    ) -> Self {
        Self {
            func,
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: shared_mem,
            kernel_params_bytes: Vec::new(),
            kernel_param_offsets: Vec::new(),
        }
    }

    pub fn push_param<T: Copy + 'static>(&mut self, value: &T) {
        let offset = self.kernel_params_bytes.len();
        let align = std::mem::align_of::<T>();
        let padding = (align - (offset % align)) % align;
        self.kernel_params_bytes.resize(offset + padding, 0);
        let aligned_offset = self.kernel_params_bytes.len();
        let bytes = unsafe {
            std::slice::from_raw_parts(value as *const T as *const u8, std::mem::size_of::<T>())
        };
        self.kernel_params_bytes.extend_from_slice(bytes);
        self.kernel_param_offsets.push(aligned_offset);
    }

    pub fn to_raw(&self) -> (Vec<*mut std::ffi::c_void>, Vec<u8>) {
        let mut buf = self.kernel_params_bytes.clone();
        let ptrs: Vec<*mut std::ffi::c_void> = self
            .kernel_param_offsets
            .iter()
            .map(|&off| unsafe { buf.as_mut_ptr().add(off) as *mut std::ffi::c_void })
            .collect();
        (ptrs, buf)
    }
}

// ---------------------------------------------------------------------------
// GraphUpdateResult
// ---------------------------------------------------------------------------

/// Result of [`CudaGraphExec::update`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphUpdateResult {
    Success,
    TopologyChanged,
    NodeTypeChanged,
    FunctionChanged,
    ParameterChanged,
    NotSupported,
    Unspecified,
}

// ---------------------------------------------------------------------------
// Stream capture
// ---------------------------------------------------------------------------

pub trait CudaStreamCaptureExt {
    fn begin_capture(&self, mode: CaptureMode) -> Result<(), DriverError>;
    fn end_capture(&self) -> Result<CudaGraph, DriverError>;
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

pub fn thread_exchange_capture_mode(mode: CaptureMode) -> Result<CaptureMode, DriverError> {
    let mut raw = mode.to_cuda();
    unsafe {
        cuda_bindings::cuThreadExchangeStreamCaptureMode(&mut raw).result()?;
    }
    CaptureMode::from_cuda(raw)
        .ok_or_else(|| DriverError(cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE))
}

pub struct CaptureModeGuard {
    prev: CaptureMode,
}

impl CaptureModeGuard {
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

// ---------------------------------------------------------------------------
// CachedGraphExec — auto-warmup graph execution
// ---------------------------------------------------------------------------

/// Strategy for how a repeated GPU workload should be executed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphStrategy {
    /// Always execute directly (no graph capture).
    Direct,
    /// Auto-detect: first run is direct, second run with no change triggers
    /// capture. Subsequent runs use the cached graph.
    AutoCapture,
    /// Use a pre-built graph from the start.
    Prebuilt,
}

/// A lazily-captured CUDA graph with automatic warmup and change detection.
///
/// Inspired by llama.cpp's CUDA graph pattern:
/// 1. First execution always runs directly (warmup).
/// 2. If the workload is stable (no property changes), the second execution
///    triggers graph capture.
/// 3. Subsequent executions launch the captured graph.
/// 4. If properties change (data pointers, shapes, etc.), the graph is
///    invalidated and re-captured.
pub struct CachedGraphExec {
    strategy: GraphStrategy,
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    /// The captured graph (if any).
    graph: Option<CudaGraph>,
    /// The instantiated executable (if captured).
    exec: Option<CudaGraphExec>,
    /// Hash of the last stable property set.
    property_hash: u64,
    /// How many times we've seen the current property hash.
    stable_count: u32,
}

impl CachedGraphExec {
    /// Creates a new cached graph executor.
    pub fn new(ctx: &Arc<CudaContext>, strategy: GraphStrategy) -> Self {
        Self {
            strategy,
            ctx: ctx.clone(),
            graph: None,
            exec: None,
            property_hash: 0,
            stable_count: 0,
        }
    }

    /// Records property information that determines graph identity.
    /// Changing any of these values invalidates a captured graph.
    pub fn set_properties(&mut self, data_ptrs: &[*const std::ffi::c_void], shapes: &[u64]) {
        let mut hash: u64 = 0x9ae16a3b2f90404f; // FNV offset basis (64-bit)
        for &ptr in data_ptrs {
            hash = hash.wrapping_mul(0x100000001b3).wrapping_add(ptr as u64);
        }
        for &shape in shapes {
            hash = hash.wrapping_mul(0x100000001b3).wrapping_add(shape);
        }
        if hash != self.property_hash {
            self.property_hash = hash;
            self.stable_count = 0;
            self.invalidate();
        }
    }

    /// Returns the strategy in use.
    pub fn strategy(&self) -> GraphStrategy {
        self.strategy
    }

    /// Returns true if a captured graph is ready for launch.
    pub fn has_cached_graph(&self) -> bool {
        self.exec.is_some()
    }

    /// Invalidates any captured graph, forcing re-capture on the next launch.
    pub fn invalidate(&mut self) {
        self.exec = None;
        self.graph = None;
    }

    /// Executes a workload using the configured strategy.
    ///
    /// - `capture_fn`: Called with the stream in capture mode to record
    ///   operations. Must NOT synchronize the stream.
    /// - `execute_fn`: Called with the stream to directly execute operations.
    ///   Used during warmup and when graphs are disabled.
    pub fn launch_or_capture<F, E>(
        &mut self,
        stream: &CudaStream,
        capture_fn: F,
        execute_fn: E,
    ) -> Result<(), DriverError>
    where
        F: FnOnce(&CudaStream) -> Result<(), DriverError>,
        E: FnOnce(&CudaStream) -> Result<(), DriverError>,
    {
        match self.strategy {
            GraphStrategy::Direct => execute_fn(stream),
            GraphStrategy::Prebuilt => {
                if let Some(ref exec) = self.exec {
                    exec.launch(stream)
                } else {
                    // No prebuilt graph provided; fall back to direct.
                    execute_fn(stream)
                }
            }
            GraphStrategy::AutoCapture => {
                if let Some(ref exec) = self.exec {
                    if self.stable_count >= 2 {
                        // Update the executable if the graph was re-captured.
                        return exec.launch(stream);
                    }
                }

                self.stable_count += 1;

                if self.stable_count < 2 {
                    // Warmup: execute directly.
                    return execute_fn(stream);
                }

                // Second stable run: capture the graph.
                self.graph = None;
                self.exec = None;

                stream.begin_capture(CaptureMode::Relaxed)?;
                capture_fn(stream)?;
                let graph = stream.end_capture()?;
                let exec = graph.instantiate()?;
                self.graph = Some(graph);
                self.exec = Some(exec);

                // Launch the freshly instantiated graph.
                if let Some(ref exec) = self.exec {
                    exec.launch(stream)
                } else {
                    Err(DriverError(
                        cuda_bindings::cudaError_enum_CUDA_ERROR_INVALID_VALUE,
                    ))
                }
            }
        }
    }

    /// Manually sets a prebuilt graph executable (for [`GraphStrategy::Prebuilt`]).
    pub fn set_prebuilt(&mut self, graph: CudaGraph, exec: CudaGraphExec) {
        self.graph = Some(graph);
        self.exec = Some(exec);
    }
}
