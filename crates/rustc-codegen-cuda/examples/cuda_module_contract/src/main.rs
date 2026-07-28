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

    /// Size requirements: the generated checked launchers prove every
    /// `requires` relation on the CPU before marshalling, so an undersized
    /// buffer becomes a typed `LaunchContractError` instead of a device
    /// fault. Evaluation is overflow-safe: operands widen to u64 and the
    /// arithmetic uses checked ops. The `_unchecked` escape hatch skips
    /// these checks just as it skips the geometry checks.
    #[kernel]
    #[launch_bounds(128)]
    #[launch_contract(
        domain = 1,
        block = (128, 1, 1),
        requires = (input.len() >= n * stride, output.len() >= n),
    )]
    pub fn strided_scale(n: usize, stride: usize, input: &[f32], mut output: DisjointSlice<f32>) {
        let index = thread::index_1d();
        let i = index.get();
        if i < n
            && let Some(out) = output.get_mut(index)
        {
            *out = input[i * stride] * 2.0;
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

    // --- Size requirements (`requires`) ---
    let n: usize = 256;
    let stride: usize = 2;
    let strided_input: Vec<f32> = (0..n * stride).map(|i| i as f32).collect();
    let strided_input_dev = DeviceBuffer::from_host(&stream, &strided_input)?;
    let mut strided_output_dev = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let strided_launch =
        module.prepare_strided_scale(LaunchConfig1D::new((n as u32).div_ceil(128), 128, 0))?;

    // (a) Buffers satisfying every relation launch normally.
    module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    )?;
    let strided_output = strided_output_dev.to_host_vec(&stream)?;
    assert!(
        strided_output
            .iter()
            .enumerate()
            .all(|(i, &value)| value == (i * stride) as f32 * 2.0),
        "strided scale produced an unexpected value",
    );

    // (b) An undersized buffer fails fast on the CPU: the launcher returns a
    // typed error carrying the violated relation's source text and both
    // evaluated sides, and nothing reaches the GPU.
    let undersized_dev = DeviceBuffer::from_host(&stream, &strided_input[..64])?;
    let violation = module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &undersized_dev,
        &mut strided_output_dev,
    );
    match violation {
        Err(
            error @ cuda_core::LaunchContractError::SizeRequirementViolated {
                relation,
                lhs,
                rhs,
                ..
            },
        ) => {
            println!("rejected undersized launch on the CPU: {error}");
            assert_eq!(relation, "input.len() >= n * stride");
            assert_eq!(lhs, 64);
            assert_eq!(rhs, 512);
        }
        other => panic!("expected SizeRequirementViolated, got {other:?}"),
    }

    // (c) Relation arithmetic is overflow-safe: an operand product leaving
    // the u64 range is its own typed error, not a wrapped comparison.
    let overflow = module.strided_scale(
        &stream,
        &strided_launch,
        usize::MAX,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    );
    match overflow {
        Err(error @ cuda_core::LaunchContractError::SizeRequirementOverflow { relation, .. }) => {
            println!("rejected overflowing relation on the CPU: {error}");
            assert_eq!(relation, "input.len() >= n * stride");
        }
        other => panic!("expected SizeRequirementOverflow, got {other:?}"),
    }

    // The two rejected launches left the stream healthy; a valid launch
    // still succeeds afterwards.
    module.strided_scale(
        &stream,
        &strided_launch,
        n,
        stride,
        &strided_input_dev,
        &mut strided_output_dev,
    )?;
    stream.synchronize()?;

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

    // A kernel declaring an exact `block` carries that shape into PTX as
    // `.reqntid`, which the driver enforces on every axis. Every kernel in
    // this example declares one, including the generic and the explicitly
    // instantiated kernels whose contracts expand before `#[kernel]` and
    // reach the entry wrapper through marker forwarding.
    for (entry, geometry) in [
        ("aligned_dynamic_shared", ".reqntid 256, 1, 1"),
        ("mixed_abi", ".reqntid 256, 1, 1"),
        ("strided_scale", ".reqntid 128, 1, 1"),
        ("helper_contract_32", ".reqntid 32, 1, 1"),
        ("helper_contract_256", ".reqntid 32, 1, 1"),
        ("explicit_aligned_u32", ".reqntid 32, 1, 1"),
    ] {
        verify_entry_geometry(&ptx, &format!(".visible .entry {entry}("), entry, geometry)?;
    }

    verify_entry_geometry(
        &ptx,
        ".visible .entry generic_aligned_TID_",
        "generic_aligned specialization",
        ".reqntid 64, 1, 1",
    )?;

    println!("SUCCESS: prepared-launch PTX contract verified");
    Ok(())
}

/// Assert one entry's launch geometry, and that it declares exactly one of the
/// two mutually exclusive directives.
///
/// ptxas rejects an entry carrying both `.maxntid` and `.reqntid`
/// ("Conflicting directives: .maxntid and .reqntid cannot both be specified"),
/// so a kernel with an exact block emits `.reqntid` in place of the thread
/// maximum. Every kernel here declares `#[launch_bounds]`, so this is the case
/// that would regress if the exporter stopped suppressing one of them.
fn verify_entry_geometry(
    ptx: &str,
    anchor: &str,
    entry: &str,
    expected: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let start = ptx
        .find(anchor)
        .ok_or_else(|| format!("missing PTX entry {entry}"))?;
    let rest = &ptx[start..];
    let end = rest[1..]
        .find(".visible .entry ")
        .map_or(rest.len(), |offset| offset + 1);
    let body = &rest[..end];

    if !body.contains(expected) {
        return Err(format!("PTX entry {entry} lost its launch geometry `{expected}`").into());
    }
    if body.contains(".maxntid") && body.contains(".reqntid") {
        return Err(format!(
            "PTX entry {entry} declares both .maxntid and .reqntid, which ptxas rejects"
        )
        .into());
    }

    Ok(())
}
