/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![allow(dead_code)]

use cuda_core::{CudaStream, DeviceBuffer, LaunchConfig};
#[cfg(feature = "async")]
use cuda_host::cuda_async::device_box::DeviceBox;
#[cfg(feature = "async")]
use cuda_host::cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_module;
use cuda_macros::{cooperative_launch, kernel};

#[cfg(feature = "async")]
type TwoF32Buffers = (DeviceBox<[f32]>, DeviceBox<[f32]>);

#[repr(C)]
#[derive(Clone, Copy)]
struct AffineParams {
    scale: f32,
    bias: f32,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn scalar_args(
        scale: f32,
        params: AffineParams,
        raw: *const f32,
        input: &[f32],
        output: &mut [f32],
    ) {
        let _ = (scale, params, raw, input, output);
    }

    #[kernel]
    pub fn copy_closure<F: Fn(u32) -> u32 + Copy>(op: F, output: &mut [u32]) {
        let _ = (op, output);
    }

    #[kernel]
    pub unsafe fn unsafe_raw_pointer(raw: *mut f32) {
        let _ = raw;
    }

    /// `#[cooperative_launch]` routes every generated launch method through
    /// the cooperative driver entry points; this kernel pins that the
    /// generated sync, async, and owned-async methods still typecheck.
    #[kernel]
    #[cooperative_launch]
    pub fn cooperative_grid_sync(output: &mut [u32]) {
        let _ = output;
    }
}

#[cfg(feature = "async")]
fn assert_unit_operation<O: DeviceOperation<Output = ()>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_two_f32_buffers<O: DeviceOperation<Output = TwoF32Buffers>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_u32_buffer<O: DeviceOperation<Output = DeviceBox<[u32]>>>(op: O) {
    let _ = op;
}

#[cfg(feature = "async")]
fn assert_owned_unit<O: DeviceOperation<Output = ()>>(op: O) {
    let _ = op;
}

fn generated_methods_accept_kernel_scalar_types(
    module: &kernels::LoadedModule,
    stream: &CudaStream,
    config: LaunchConfig,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    output_u32: &mut DeviceBuffer<u32>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();

    module.scalar_args(stream, config, 2.0, params, raw, input, output)?;

    let offset = 5u32;
    let op = move |x: u32| x + offset;
    module.copy_closure(stream, config, op, output_u32)?;

    let raw_mut = core::ptr::null_mut::<f32>();
    unsafe {
        module.unsafe_raw_pointer(stream, config, raw_mut)?;
    }

    module.cooperative_grid_sync(stream, config, output_u32)?;

    Ok(())
}

#[cfg(feature = "async")]
fn generated_async_methods_accept_borrowed_buffers(
    module: &kernels::LoadedModule,
    config: LaunchConfig,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    async_input: &DeviceBox<[f32]>,
    async_output: &mut DeviceBox<[f32]>,
    async_output_u32: &mut DeviceBox<[u32]>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();
    let raw_mut = core::ptr::null_mut::<f32>();

    let launch = module.scalar_args_async(config, 2.0, params, raw, input, output)?;
    assert_unit_operation(launch);

    let launch = module.scalar_args_async(config, 2.0, params, raw, async_input, async_output)?;
    assert_unit_operation(launch);

    let offset = 5u32;
    let offset_ref = &offset;
    let op = |x: u32| x + *offset_ref;
    let launch = module.copy_closure_async(config, op, async_output_u32)?;
    assert_unit_operation(launch);

    unsafe {
        let launch = module.unsafe_raw_pointer_async(config, raw_mut)?;
        assert_unit_operation(launch);
    }

    let launch = module.cooperative_grid_sync_async(config, async_output_u32)?;
    assert_unit_operation(launch);

    Ok(())
}

#[cfg(feature = "async")]
fn generated_owned_async_methods_accept_owned_buffers(
    module: &kernels::LoadedModule,
    config: LaunchConfig,
    async_input: DeviceBox<[f32]>,
    async_output: DeviceBox<[f32]>,
    async_output_u32: DeviceBox<[u32]>,
    async_coop_output_u32: DeviceBox<[u32]>,
) -> Result<(), cuda_core::DriverError> {
    let params = AffineParams {
        scale: 2.0,
        bias: 1.0,
    };
    let raw = core::ptr::null::<f32>();
    let raw_mut = core::ptr::null_mut::<f32>();

    let launch: cuda_host::OwnedAsyncKernelLaunch<TwoF32Buffers> =
        module.scalar_args_async_owned(config, 2.0, params, raw, async_input, async_output)?;
    assert_owned_two_f32_buffers(launch);

    let offset = 5u32;
    let op = move |x: u32| x + offset;
    let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
        module.copy_closure_async_owned(config, op, async_output_u32)?;
    assert_owned_u32_buffer(launch);

    unsafe {
        let launch: cuda_host::OwnedAsyncKernelLaunch<()> =
            module.unsafe_raw_pointer_async_owned(config, raw_mut)?;
        assert_owned_unit(launch);
    }

    let launch: cuda_host::OwnedAsyncKernelLaunch<DeviceBox<[u32]>> =
        module.cooperative_grid_sync_async_owned(config, async_coop_output_u32)?;
    assert_owned_u32_buffer(launch);

    Ok(())
}

#[test]
fn generated_cuda_module_api_typechecks() {
    let _ = generated_methods_accept_kernel_scalar_types;
    #[cfg(feature = "async")]
    let _ = generated_async_methods_accept_borrowed_buffers;
    #[cfg(feature = "async")]
    let _ = generated_owned_async_methods_accept_owned_buffers;
}

// =============================================================================
// PTX naming contract
//
// These tests pin down the shape of the host-side `GenericCudaKernel::ptx_name`
// output. The backend's `compute_kernel_export_name` in
// `crates/rustc-codegen-cuda/src/collector.rs` follows the same scheme, so
// any drift here is the canary that the two sides have diverged again.
//
// On-wire shape: `<base>_TID_<hex32>`. The single `<hex32>` is the hash of
// the tuple of generic args, not one hash per arg — so the length stays
// constant regardless of generic arity.
// =============================================================================

use cuda_host::GenericCudaKernel;

fn is_lowercase_hex_32(s: &str) -> bool {
    s.len() == 32
        && s.chars()
            .all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

fn split_tid_name<'a>(name: &'a str, base: &str) -> &'a str {
    let expected_prefix = format!("{base}_TID_");
    name.strip_prefix(&expected_prefix)
        .unwrap_or_else(|| panic!("expected `{name}` to start with `{expected_prefix}`"))
}

#[test]
fn ptx_name_for_closure_generic_matches_tid_scheme() {
    let offset = 5u32;
    let op = move |x: u32| x + offset;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        <kernels::__copy_closure_CudaKernel<F> as GenericCudaKernel>::ptx_name()
    }

    let name = name_for(op);
    let hex = split_tid_name(name, "copy_closure");
    assert!(
        is_lowercase_hex_32(hex),
        "expected `<base>_TID_<32hex>`; got `{name}` (suffix `{hex}`)"
    );
}

#[test]
fn ptx_name_is_stable_per_closure_type() {
    let offset = 7u32;
    let op = move |x: u32| x + offset;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        <kernels::__copy_closure_CudaKernel<F> as GenericCudaKernel>::ptx_name()
    }
    let a = name_for(op);
    let b = name_for(op);
    assert_eq!(a, b, "same closure type must produce the same PTX name");
}

#[test]
fn ptx_name_separates_distinct_closure_types() {
    let factor = 2u32;
    let op1 = move |x: u32| x + factor;
    let op2 = move |x: u32| x * factor;
    fn name_for<F: Fn(u32) -> u32 + Copy>(_f: F) -> &'static str {
        <kernels::__copy_closure_CudaKernel<F> as GenericCudaKernel>::ptx_name()
    }
    let n1 = name_for(op1);
    let n2 = name_for(op2);
    assert_ne!(
        n1, n2,
        "two distinct closure literals must produce different PTX names ({n1} vs {n2})"
    );
}
