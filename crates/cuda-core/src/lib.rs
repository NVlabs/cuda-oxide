/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Safe RAII wrappers around the CUDA driver API.
//!
//! This crate provides Rust-idiomatic, RAII-managed wrappers over the low-level CUDA
//! driver API (`cuInit`, `cuCtx*`, `cuStream*`, `cuEvent*`, `cuModule*`, `cuMem*`).
//! All GPU resources are released automatically on [`Drop`], and error codes are
//! propagated as [`DriverError`] values.
//!
//! # Architecture
//!
//! - [`CudaContext`] -- retains the device primary context and tracks per-context
//!   state (stream count, error accumulation).
//! - [`CudaStream`] -- non-blocking stream with fork/join parallelism and host
//!   callback support for bridging to async Rust.
//! - [`CudaEvent`] -- lightweight synchronization primitive for inter-stream
//!   ordering and elapsed-time measurement.
//! - [`CudaModule`] / [`CudaFunction`] -- PTX/cubin loading and kernel handle
//!   extraction.
//! - [`PinnedHostBuffer`] -- page-locked host memory for CUDA transfers.
//! - [`LaunchConfig`] -- grid/block dimension helper.
//! - [`memory`] -- free functions for device allocation, transfer, and memset
//!   (both stream-ordered async and synchronous variants).
//!
//! # Context binding
//!
//! Most driver calls require a CUDA context bound to the calling thread. Methods
//! on the wrapper types call [`CudaContext::bind_to_thread`] internally so the
//! caller does not need to manage `cuCtxSetCurrent` manually.
//!
//! # Re-exports
//!
//! Raw bindings are available as [`sys`] for any driver entry point not yet
//! wrapped.

#![feature(f16)]

/// CUDA context management (primary context, RAII).
pub mod context;
/// Owning device memory buffer with host-device transfer helpers.
pub mod device_buffer;
/// Embedded device artifact discovery and CUDA module loading.
pub mod embedded;
/// CUDA driver error types and result conversion.
pub mod error;
/// CUDA event management (timing, synchronization).
pub mod event;
/// Kernel launch configuration helpers.
pub mod launch;
/// Device memory allocation and transfer operations.
pub mod memory;
/// CUDA module and function management (PTX/cubin loading).
pub mod module;
/// Peer-to-peer (P2P) access between GPU contexts.
pub mod peer;
/// Page-locked host memory for CUDA transfers.
pub mod pinned_host_buffer;
/// CUDA stream management (RAII, host callbacks, fork/join).
pub mod stream;
/// CUDA Virtual Memory Management (VMM) for physical alloc, VA reservation, and mapping.
pub mod vmm;

pub use context::CudaContext;
/// Raw CUDA driver bindings re-exported for direct access when needed.
pub use cuda_bindings as sys;
pub use device_buffer::{DeviceBuffer, DeviceCopy};
pub use embedded::{EmbeddedModule, EmbeddedModuleError};
pub use error::{DriverError, IntoResult};
pub use event::CudaEvent;
pub use launch::LaunchConfig;
pub use module::{ConstantHandle, CudaFunction, CudaModule};
pub use pinned_host_buffer::PinnedHostBuffer;
pub use stream::CudaStream;

use std::ffi::c_uint;

/// Initializes the CUDA driver API. Must be called before any other driver API
/// function. Safe to call multiple times; subsequent calls are no-ops.
///
/// `flags` is reserved by CUDA and must currently be `0`.
///
/// [`CudaContext::new`] calls this automatically, so explicit invocation is only
/// needed when using raw bindings via [`sys`].
///
/// # Safety
///
/// - A CUDA-capable device and a compatible driver must be installed.
/// - Must not be called from a GPU callback (`cuLaunchHostFunc`).
pub unsafe fn init(flags: c_uint) -> Result<(), DriverError> {
    unsafe { cuda_bindings::cuInit(flags) }.result()
}

/// Low-level wrapper around `cuLaunchKernel`.
///
/// The launch is **asynchronous** with respect to the host: this function
/// returns as soon as the launch is enqueued. Use stream or event
/// synchronization to wait for completion.
///
/// `grid_dim` and `block_dim` are `(x, y, z)` tuples specifying the grid and
/// block dimensions respectively. `shared_mem_bytes` reserves per-block dynamic
/// shared memory (in bytes).
///
/// `kernel_params` is a mutable slice of pointers, one per kernel argument,
/// each pointing to the argument value. The `extra` parameter is set to null
/// (use raw [`sys::cuLaunchKernel`] directly if you need more advanced launch
/// modes).
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_on_stream`] for normal host-side code: it takes a typed
/// [`CudaStream`], binds that stream's owning context to the calling thread,
/// and then forwards to this raw entry point.
///
/// # Safety
///
/// - `func` must be a valid `CUfunction` obtained from a loaded module.
/// - `stream` must be a valid `CUstream` belonging to the current context, or
///   null for the default stream.
/// - Each element of `kernel_params` must point to a region of memory that
///   matches the corresponding kernel parameter in size and alignment.
/// - The pointed-to argument values must remain valid until this function
///   returns, because the driver reads them during launch submission.
/// - The grid and block dimensions must not exceed device limits.
/// - The calling thread must already have the context that owns both `func` and
///   `stream` bound as its current context.
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernel` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    unsafe {
        cuda_bindings::cuLaunchKernel(
            func,
            grid_dim.0,
            grid_dim.1,
            grid_dim.2,
            block_dim.0,
            block_dim.1,
            block_dim.2,
            shared_mem_bytes,
            stream,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a CUDA kernel on a specific [`CudaStream`], binding its context first.
///
/// This is the usual host-side helper for `cuda_launch!` and async launches.
/// It ensures the stream's owning context is current before calling the raw
/// [`launch_kernel`] entry point, so callers do not need to manually call
/// [`CudaContext::bind_to_thread`] before every launch.
///
/// Unlike [`launch_kernel`], this helper works with typed wrappers rather than
/// raw driver handles. It is therefore the preferred API whenever you already
/// have a [`CudaFunction`] and [`CudaStream`].
///
/// # Safety
///
/// - `func` must refer to a kernel loaded from the same CUDA context that owns
///   `stream`.
/// - Each element of `kernel_params` must point to a region of memory that
///   matches the corresponding kernel parameter in size and alignment.
/// - The pointed-to argument values must remain valid until this function
///   returns.
/// - The grid and block dimensions must not exceed device limits.
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernel` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_on_stream(
    func: &CudaFunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &CudaStream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx`.
///
/// This is the cluster-aware launch path. It builds a `CUlaunchConfig` with a
/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` attribute set to `cluster_dim`.
/// Required for thread-block cluster launches on sm_90+ (Hopper / Blackwell).
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_ex_on_stream`] in normal host-side code so the correct
/// stream context is made current automatically.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel`], plus:
/// - Each component of `cluster_dim` must divide the corresponding `grid_dim`
///   component.
/// - The total cluster size must not exceed the device maximum.
/// - The device must support compute capability 9.0 or higher.
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_ex(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // clusterDim is three u32 fields (x, y, z) at offset 0 within the value union.
    let mut cluster_attr: cuda_bindings::CUlaunchAttribute_st = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut cluster_attr as *mut _ as *mut u8;
        // id at offset 0
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
        // clusterDim.x/y/z at offsets 8, 12, 16
        let dim_ptr = base.add(8) as *mut u32;
        dim_ptr.write(cluster_dim.0);
        dim_ptr.add(1).write(cluster_dim.1);
        dim_ptr.add(2).write(cluster_dim.2);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &mut cluster_attr,
        numAttrs: 1,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a CUDA kernel with extended configuration on a specific stream,
/// binding the stream's owning context first.
///
/// This is the cluster-aware counterpart to [`launch_kernel_on_stream`]. It
/// binds `stream.context()` to the calling thread, then forwards to the raw
/// [`launch_kernel_ex`] helper.
///
/// # Safety
///
/// - `func` must refer to a kernel loaded from the same CUDA context that owns
///   `stream`.
/// - Each element of `kernel_params` must point to a region of memory that
///   matches the corresponding kernel parameter in size and alignment.
/// - The pointed-to argument values must remain valid until this function
///   returns.
/// - The grid, block, and cluster dimensions must satisfy the device limits and
///   CUDA cluster-launch requirements.
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_ex_on_stream(
    func: &CudaFunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: &CudaStream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_ex(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            cluster_dim,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx` with the
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` flag set.
///
/// A *cooperative* launch guarantees that every block in the grid is
/// co-resident on the device, which is the precondition for grid-wide
/// barriers like `cuda_device::grid::sync()`. The CUDA driver also
/// populates PTX environment registers `%envreg1` / `%envreg2` with the
/// pointer to the per-launch grid workspace; the device-side barrier
/// implementation reads those registers to find the shared counter.
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_cooperative_on_stream`] in normal host-side code so the
/// correct stream context is made current automatically.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel`], plus:
/// - The device must support cooperative launch (`CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`).
/// - The grid must fit in the maximum number of resident blocks for this
///   kernel as reported by `cuOccupancyMaxActiveBlocksPerMultiprocessor`
///   (otherwise `cuLaunchKernelEx` returns
///   `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`).
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_cooperative(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // For the COOPERATIVE attribute the value union holds a single `int cooperative`
    // at offset 0 — set to 1 to enable, 0 to disable.
    let mut coop_attr: cuda_bindings::CUlaunchAttribute_st = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut coop_attr as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE);
        let val_ptr = base.add(8) as *mut i32;
        val_ptr.write(1);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: &mut coop_attr,
        numAttrs: 1,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a cooperative CUDA kernel on a specific stream, binding the
/// stream's owning context first.
///
/// This is the cooperative-launch counterpart to [`launch_kernel_on_stream`].
/// It binds `stream.context()` to the calling thread, then forwards to the
/// raw [`launch_kernel_cooperative`] helper.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel_cooperative`].
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_cooperative_on_stream(
    func: &CudaFunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    stream: &CudaStream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_cooperative(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            stream.cu_stream(),
            kernel_params,
        )
    }
}

/// Low-level wrapper around `cuLaunchKernelEx` with **both** the
/// `CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION` and
/// `CU_LAUNCH_ATTRIBUTE_COOPERATIVE` attributes set.
///
/// `cuLaunchKernelEx` takes an array of launch attributes, so a single call
/// can request thread-block clusters ([`launch_kernel_ex`]) and a
/// cooperative launch ([`launch_kernel_cooperative`]) at the same time.
/// This is the path used when a `#[cuda_module]` kernel carries both
/// `#[cluster_launch(...)]` and `#[cooperative_launch]`.
///
/// This helper performs **no context binding**. Prefer
/// [`launch_kernel_ex_cooperative_on_stream`] in normal host-side code so
/// the correct stream context is made current automatically.
///
/// # Safety
///
/// The combined preconditions of [`launch_kernel_ex`] (cluster dimensions
/// must divide the grid, sm_90+) and [`launch_kernel_cooperative`] (the
/// device must support cooperative launch and the whole grid must be
/// co-resident).
///
/// # Errors
///
/// Returns the CUDA driver error produced by `cuLaunchKernelEx` if launch
/// submission fails.
#[inline]
pub unsafe fn launch_kernel_ex_cooperative(
    func: cuda_bindings::CUfunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: cuda_bindings::CUstream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    // CUlaunchAttribute_st is opaque (see cuda-bindings/build.rs) for CUDA 13.2+
    // compatibility. C layout: { id: u32 @ 0, pad: [u8;4] @ 4, value: union @ 8 }.
    // attrs[0]: clusterDim — three u32 fields (x, y, z) at offset 0 of the union.
    // attrs[1]: cooperative — a single `int` at offset 0 of the union; 1 = enabled.
    let mut attrs: [cuda_bindings::CUlaunchAttribute_st; 2] = unsafe { std::mem::zeroed() };
    unsafe {
        let base = &mut attrs[0] as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION);
        let dim_ptr = base.add(8) as *mut u32;
        dim_ptr.write(cluster_dim.0);
        dim_ptr.add(1).write(cluster_dim.1);
        dim_ptr.add(2).write(cluster_dim.2);

        let base = &mut attrs[1] as *mut _ as *mut u8;
        (base as *mut u32)
            .write(cuda_bindings::CUlaunchAttributeID_enum_CU_LAUNCH_ATTRIBUTE_COOPERATIVE);
        let val_ptr = base.add(8) as *mut i32;
        val_ptr.write(1);
    }

    let config = cuda_bindings::CUlaunchConfig_st {
        gridDimX: grid_dim.0,
        gridDimY: grid_dim.1,
        gridDimZ: grid_dim.2,
        blockDimX: block_dim.0,
        blockDimY: block_dim.1,
        blockDimZ: block_dim.2,
        sharedMemBytes: shared_mem_bytes,
        hStream: stream,
        attrs: attrs.as_mut_ptr(),
        numAttrs: 2,
    };

    unsafe {
        cuda_bindings::cuLaunchKernelEx(
            &config,
            func,
            kernel_params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
    .result()
}

/// Launches a cooperative CUDA kernel with cluster dimensions on a specific
/// stream, binding the stream's owning context first.
///
/// This is the cluster-plus-cooperative counterpart to
/// [`launch_kernel_on_stream`]. It binds `stream.context()` to the calling
/// thread, then forwards to the raw [`launch_kernel_ex_cooperative`] helper.
///
/// # Safety
///
/// Same preconditions as [`launch_kernel_ex_cooperative`].
///
/// # Errors
///
/// Returns an error if binding `stream.context()` fails or if the underlying
/// `cuLaunchKernelEx` call rejects the launch.
#[inline]
pub unsafe fn launch_kernel_ex_cooperative_on_stream(
    func: &CudaFunction,
    grid_dim: (u32, u32, u32),
    block_dim: (u32, u32, u32),
    shared_mem_bytes: u32,
    cluster_dim: (u32, u32, u32),
    stream: &CudaStream,
    kernel_params: &mut [*mut std::ffi::c_void],
) -> Result<(), DriverError> {
    stream.context().bind_to_thread()?;
    unsafe {
        launch_kernel_ex_cooperative(
            func.cu_function(),
            grid_dim,
            block_dim,
            shared_mem_bytes,
            cluster_dim,
            stream.cu_stream(),
            kernel_params,
        )
    }
}
