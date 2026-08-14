/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Kernel-boundary ABI regression coverage for `#[repr(transparent)]` scalar ADTs.
//!
//! The device entry must expose the underlying scalar/pointer parameter rather
//! than an aggregate `.param .b8[...]`. An ordinary one-field struct is kept as
//! the negative control and must remain aggregate at the kernel boundary.

use core::marker::PhantomData;

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{Uniform, cuda_module, kernel};

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Scalar(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Pointer(pub *const u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Marked(pub u32, pub PhantomData<()>);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Inner(pub u32);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct Outer(pub Inner);

#[derive(Clone, Copy)]
pub struct Ordinary(pub u32);

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub unsafe fn scalar(value: Scalar, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }

    #[kernel]
    pub unsafe fn pointer(value: Pointer, out: *mut u32) {
        // SAFETY: the host launch passes one readable u32 through `value` and
        // one writable u32 through `out`, both live through synchronization.
        unsafe { out.write(value.0.read()) };
    }

    #[kernel]
    pub unsafe fn marked(value: Marked, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }

    #[kernel]
    pub unsafe fn nested(value: Outer, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0.0) };
    }

    #[kernel]
    pub unsafe fn uniform(value: Uniform<u32>, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.get()) };
    }

    #[kernel]
    pub unsafe fn ordinary(value: Ordinary, out: *mut u32) {
        // SAFETY: the host launch passes one writable u32 in `out`.
        unsafe { out.write(value.0) };
    }
}

fn entry_header<'a>(ptx: &'a str, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let marker = format!(".visible .entry {name}(");
    let start = ptx
        .find(&marker)
        .ok_or_else(|| format!("missing PTX entry `{name}`"))?;
    let rest = &ptx[start..];
    let end = rest
        .find('{')
        .ok_or_else(|| format!("unterminated PTX entry header `{name}`"))?;
    Ok(&rest[..end])
}

fn require_scalar_header(
    ptx: &str,
    name: &str,
    scalar_token: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = entry_header(ptx, name)?;
    if header.contains(".b8") {
        return Err(
            format!("kernel `{name}` still exposes an aggregate PTX parameter:\n{header}").into(),
        );
    }
    if !header.contains(scalar_token) {
        return Err(format!(
            "kernel `{name}` does not expose expected `{scalar_token}` parameter:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn verify_generated_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("repr_transparent_abi.ptx");
    let ptx = std::fs::read_to_string(&path)?;

    require_scalar_header(&ptx, "scalar", ".param .u32")?;
    require_scalar_header(&ptx, "pointer", ".param .u64")?;
    require_scalar_header(&ptx, "marked", ".param .u32")?;
    require_scalar_header(&ptx, "nested", ".param .u32")?;
    require_scalar_header(&ptx, "uniform", ".param .u32")?;

    let ordinary = entry_header(&ptx, "ordinary")?;
    if !ordinary.contains(".b8") {
        return Err(format!(
            "ordinary one-field struct was incorrectly scalarized at the kernel boundary:\n{ordinary}"
        )
        .into());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        verify_generated_ptx()?;
        println!("repr_transparent_abi: PASS (PTX parameter shapes)");
        return Ok(());
    }

    verify_generated_ptx()?;

    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let module = kernels::load(&context)?;
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let input = DeviceBuffer::from_host(&stream, &[17u32])?;
    let scalar_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let pointer_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let marked_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let nested_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let uniform_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;
    let ordinary_out = DeviceBuffer::<u32>::zeroed(&stream, 1)?;

    // SAFETY: every kernel launches one thread. Each output buffer owns one
    // writable u32, and `input` owns the readable u32 used by `pointer`.
    unsafe {
        module.scalar(
            &stream,
            config,
            Scalar(11),
            scalar_out.cu_deviceptr() as *mut u32,
        )?;
        module.pointer(
            &stream,
            config,
            Pointer(input.cu_deviceptr() as *const u32),
            pointer_out.cu_deviceptr() as *mut u32,
        )?;
        module.marked(
            &stream,
            config,
            Marked(23, PhantomData),
            marked_out.cu_deviceptr() as *mut u32,
        )?;
        module.nested(
            &stream,
            config,
            Outer(Inner(29)),
            nested_out.cu_deviceptr() as *mut u32,
        )?;
        // `#[cuda_module]` intentionally maps `Uniform<u32>` to a bare host
        // `u32`; the device type remains the proof-carrying transparent ADT.
        module.uniform(
            &stream,
            config,
            31u32,
            uniform_out.cu_deviceptr() as *mut u32,
        )?;
        module.ordinary(
            &stream,
            config,
            Ordinary(37),
            ordinary_out.cu_deviceptr() as *mut u32,
        )?;
    }

    assert_eq!(scalar_out.to_host_vec(&stream)?, [11]);
    assert_eq!(pointer_out.to_host_vec(&stream)?, [17]);
    assert_eq!(marked_out.to_host_vec(&stream)?, [23]);
    assert_eq!(nested_out.to_host_vec(&stream)?, [29]);
    assert_eq!(uniform_out.to_host_vec(&stream)?, [31]);
    assert_eq!(ordinary_out.to_host_vec(&stream)?, [37]);

    println!("repr_transparent_abi: PASS (runtime values and PTX parameter shapes)");
    Ok(())
}
