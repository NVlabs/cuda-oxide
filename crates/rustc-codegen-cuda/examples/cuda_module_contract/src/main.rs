/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Focused `#[cuda_module]` host ABI contract test.
//!
//! The kernel intentionally mixes common host-side argument shapes the typed
//! module macro must lower correctly: scalars, slice, raw device pointer, and
//! `DisjointSlice` output.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig1D};
use cuda_device::{
    DisjointSlice, DynamicSharedArray, cuda_module, kernel, launch_bounds, launch_contract, thread,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn ordinary_shared_owner(value: u32) {
        let shared = DynamicSharedArray::<u32, 16>::get();
        unsafe {
            core::ptr::write_volatile(shared, value);
        }
    }

    #[inline(never)]
    fn ordinary_shared_forward(value: u32) {
        ordinary_shared_owner(value);
    }

    /// Two entries share the same transitive helper. The helper's single PTX
    /// declaration must use the stronger contract from either caller.
    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 32,
    )]
    pub fn helper_contract_32(value: u32) {
        ordinary_shared_forward(value);
    }

    #[kernel]
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 256,
    )]
    pub fn helper_contract_256(value: u32) {
        ordinary_shared_forward(value);
    }

    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(domain = 1, block = (256, 1, 1), dynamic_shared = 0)]
    pub fn mixed_abi(
        scale: f32,
        bias: f32,
        extra: f32,
        input: &[f32],
        raw_offsets: *const f32,
        mut output: DisjointSlice<f32>,
    ) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(out_elem) = output.get_mut(idx) {
            let offset = unsafe { *raw_offsets.add(idx_raw) };
            *out_elem = input[idx_raw] * scale + bias + extra + offset;
        }
    }

    /// Compile-time proof that a contract alignment is merged with alignment
    /// requested by the body. The body asks for 16 bytes; the contract raises
    /// the emitted extern-shared declaration to 128 bytes.
    #[kernel]
    #[launch_bounds(256)]
    #[launch_contract(
        domain = 1,
        block = (256, 1, 1),
        dynamic_shared = 1024,
        dynamic_shared_alignment = 128,
    )]
    pub fn aligned_dynamic_shared(mut output: DisjointSlice<u8>) {
        let index = thread::index_1d();
        let linear = index.get();
        let shared = DynamicSharedArray::<u8, 16>::get_raw();
        unsafe {
            *shared.add(thread::threadIdx_x() as usize) = linear as u8;
        }
        if let Some(output) = output.get_mut(index) {
            *output = unsafe { *shared.add(thread::threadIdx_x() as usize) };
        }
    }

    /// Generic/closure pin: the prepared brand and compiler-side alignment
    /// marker must both survive monomorphization onto the exported wrapper.
    // Deliberately put both configuration attributes above #[kernel]. They
    // expand into body markers before the generic entry wrapper is generated.
    #[launch_contract(
        domain = 1,
        block = (64, 1, 1),
        dynamic_shared = 256,
        dynamic_shared_alignment = 64,
    )]
    #[launch_bounds(64)]
    #[kernel]
    pub fn generic_aligned<F: Fn(u32) -> u32 + Copy>(op: F, mut output: DisjointSlice<u32>) {
        let index = thread::index_1d();
        let linear = index.get();
        let shared = DynamicSharedArray::<u32, 16>::get();
        unsafe {
            *shared.add(thread::threadIdx_x() as usize) = op(linear as u32);
        }
        if let Some(output) = output.get_mut(index) {
            *output = unsafe { *shared.add(thread::threadIdx_x() as usize) };
        }
    }
}

/// Compile-only coverage for `#[kernel(Type)]`, the explicit-instantiation
/// form. Its concrete entry still calls a generic helper, so the entry's
/// alignment contract must propagate to the helper that owns shared memory.
mod explicit_instantiation {
    use super::*;

    // The explicit-instantiation expansion must forward pre-expanded markers
    // just like the call-site monomorphization path above.
    #[launch_bounds(32)]
    #[launch_contract(
        domain = 1,
        coordinates = u32,
        block = (32, 1, 1),
        dynamic_shared = 128,
        dynamic_shared_alignment = 32,
    )]
    #[kernel(u32, launch_context = launch_context)]
    pub fn explicit_aligned<T: Copy>(value: T) {
        let _index = thread::index_1d_u32(launch_context);
        let shared = DynamicSharedArray::<T, 8>::get();
        unsafe {
            core::ptr::write_volatile(shared, value);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::args().any(|arg| arg == "--verify-ptx") {
        return verify_launch_contract_ptx();
    }

    println!("=== cuda_module ABI Contract Test ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // SAFETY: this example has one device-code owner, and `cargo oxide` builds
    // the merged PTX set from the `kernels` module above with no conflicting
    // entry definitions.
    let module = unsafe { kernels::load(&ctx)? };

    const N: usize = 1024;
    let scale = 1.5f32;
    let bias = 2.0f32;
    let extra = 7.0f32;
    let input_host: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let offset_host: Vec<f32> = (0..N).map(|i| (i % 5) as f32).collect();

    let input_dev = DeviceBuffer::from_host(&stream, &input_host)?;
    let offset_dev = DeviceBuffer::from_host(&stream, &offset_host)?;
    let mut output_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let launch = module.prepare_mixed_abi(LaunchConfig1D::new((N as u32).div_ceil(256), 256, 0))?;

    module.mixed_abi(
        &stream,
        &launch,
        scale,
        bias,
        extra,
        &input_dev,
        offset_dev.cu_deviceptr() as *const f32,
        &mut output_dev,
    )?;

    let output = output_dev.to_host_vec(&stream)?;
    let errors = (0..N)
        .filter(|&i| {
            let expected = input_host[i] * scale + bias + extra + offset_host[i];
            (output[i] - expected).abs() > 1e-5
        })
        .count();

    assert_eq!(errors, 0, "mixed ABI kernel produced {errors} errors");

    let mut generic_output = DeviceBuffer::<u32>::zeroed(&stream, N)?;
    let add_three = |value: u32| value + 3;
    let generic_launch = module.prepare_generic_aligned_for(
        &add_three,
        LaunchConfig1D::new((N as u32).div_ceil(64), 64, 256),
    )?;
    module.generic_aligned(&stream, &generic_launch, add_three, &mut generic_output)?;
    let generic_output = generic_output.to_host_vec(&stream)?;
    assert!(
        generic_output
            .iter()
            .enumerate()
            .all(|(index, &value)| value == index as u32 + 3),
        "generic prepared launch produced an unexpected value",
    );

    println!("SUCCESS: mixed ABI typed launch passed");
    Ok(())
}

fn verify_launch_contract_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let ptx_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("cuda_module_contract.ptx");
    let ptx = std::fs::read_to_string(&ptx_path)?;
    let aligned_symbol = ".extern .shared .align 128 .b8 __dynamic_smem_aligned_dynamic_shared[];";
    if !ptx.contains(aligned_symbol) {
        return Err(format!(
            "{} does not contain the contract-enforced dynamic shared-memory alignment",
            ptx_path.display()
        )
        .into());
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 64 .b8 __dynamic_smem_")
            && line.contains("generic_aligned")
    }) {
        return Err(
            "generic launch contract alignment did not reach its PTX specialization".into(),
        );
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 32 .b8 __dynamic_smem_")
            && line.contains("explicit_aligned")
    }) {
        return Err("explicit generic instantiation alignment did not reach its PTX helper".into());
    }
    if !ptx.lines().any(|line| {
        line.contains(".extern .shared .align 256 .b8 __dynamic_smem_")
            && line.contains("ordinary_shared_owner")
    }) {
        return Err(
            "shared ordinary helper did not receive the strongest calling-kernel alignment".into(),
        );
    }

    for entry in ["aligned_dynamic_shared", "mixed_abi"] {
        let start = ptx
            .find(&format!(".visible .entry {entry}("))
            .ok_or_else(|| format!("missing PTX entry {entry}"))?;
        let rest = &ptx[start..];
        let end = rest[1..]
            .find(".visible .entry ")
            .map_or(rest.len(), |offset| offset + 1);
        if !rest[..end].contains(".maxntid 256, 1, 1") {
            return Err(format!("PTX entry {entry} lost its launch bounds").into());
        }
    }

    for entry in [
        "helper_contract_32",
        "helper_contract_256",
        "explicit_aligned_u32",
    ] {
        let start = ptx
            .find(&format!(".visible .entry {entry}("))
            .ok_or_else(|| format!("missing PTX entry {entry}"))?;
        let rest = &ptx[start..];
        let end = rest[1..]
            .find(".visible .entry ")
            .map_or(rest.len(), |offset| offset + 1);
        if !rest[..end].contains(".maxntid 32, 1, 1") {
            return Err(format!("PTX entry {entry} lost its launch bounds").into());
        }
    }

    let generic_start = ptx
        .find(".visible .entry generic_aligned_TID_")
        .ok_or("missing generic_aligned PTX specialization")?;
    let generic_rest = &ptx[generic_start..];
    let generic_end = generic_rest[1..]
        .find(".visible .entry ")
        .map_or(generic_rest.len(), |offset| offset + 1);
    if !generic_rest[..generic_end].contains(".maxntid 64, 1, 1") {
        return Err("generic_aligned PTX specialization lost its launch bounds".into());
    }

    println!("SUCCESS: prepared-launch PTX contract verified");
    Ok(())
}
