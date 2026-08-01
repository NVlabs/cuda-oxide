/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Ordinary device global static example.
//!
//! Build and run with:
//!   cargo oxide run device_global

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::kernel;
use cuda_host::cuda_module;

static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], [1.0, 2.0], [4.0, 8.0], [16.0, 32.0]];
static STATIC_NAN: f32 = f32::from_bits(0x7fc0_1234);

const STATIC_WEIGHT_PAIR: &[f32; 2] = &STATIC_WEIGHTS[2];

/// These targets are intentionally reached only through other static
/// initializers. Their materialization therefore exercises transitive device
/// global discovery rather than a direct reference from a kernel body.
static RELOCATION_TARGET_A: u32 = 0x1234_5678;
static RELOCATION_TARGET_B: u32 = 0xcafe_babe;
static RELOCATION_REFERENCE: &u32 = &RELOCATION_TARGET_A;
static RELOCATION_REFERENCES: [&u32; 2] = [&RELOCATION_TARGET_A, &RELOCATION_TARGET_B];
static INTERIOR_RELOCATION_REFERENCE: &f32 = &STATIC_WEIGHTS[2][1];

/// One-past-the-end interior pointer: const eval permits forming a pointer
/// whose addend equals the allocation size (32 bytes here). It is legal to
/// form and compare, only dereferencing it would be UB, so the translator
/// must materialize it instead of rejecting the offset.
const STATIC_WEIGHTS_END: *const [f32; 2] =
    unsafe { (&raw const STATIC_WEIGHTS as *const [f32; 2]).add(4) };

#[repr(C)]
struct PaddedStatic {
    tag: u8,
    value: u32,
}

static PADDED_STATIC: PaddedStatic = PaddedStatic {
    tag: 0xab,
    value: 0x1234_5678,
};

#[inline(never)]
fn get_static_weights() -> &'static [[f32; 2]; 4] {
    &STATIC_WEIGHTS
}

#[inline(never)]
fn get_static_weight_pair() -> &'static [f32; 2] {
    STATIC_WEIGHT_PAIR
}

#[inline(never)]
fn get_static_nan() -> &'static f32 {
    &STATIC_NAN
}

#[inline(never)]
fn get_padded_static() -> &'static PaddedStatic {
    &PADDED_STATIC
}

#[inline(never)]
fn get_padded_static_tag() -> &'static u8 {
    &PADDED_STATIC.tag
}

#[inline(never)]
fn get_padded_static_value() -> &'static u32 {
    &PADDED_STATIC.value
}

#[inline(never)]
fn get_static_weights_end() -> *const [f32; 2] {
    STATIC_WEIGHTS_END
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to a writable `u64` in device-accessible memory.
    /// The static globals `DEVICE_COUNTER` and `DEVICE_MARKER` are mutated
    /// without synchronisation; the test launches a single thread to dodge
    /// the race.
    #[kernel]
    pub unsafe fn device_global(out: *mut u64) {
        unsafe {
            DEVICE_COUNTER += 1;
            DEVICE_MARKER = 0x00C0_FFEE;
            *out = DEVICE_COUNTER ^ (DEVICE_MARKER as u64);
        }
    }

    /// Read both the base address and an interior pointer into an immutable
    /// device static.
    ///
    /// `STATIC_WEIGHT_PAIR` carries the provenance of `STATIC_WEIGHTS` plus
    /// a 16-byte addend selecting element 2.
    #[kernel]
    pub unsafe fn nonzero_static_table(out: *mut f32) {
        let weights = get_static_weights();
        let pair = get_static_weight_pair();

        unsafe {
            *out = weights[0][0] + pair[0] + pair[1];
        }
    }

    /// Preserve exact initializer bits and Rust's evaluated field offsets.
    #[kernel]
    pub unsafe fn static_initializer_edges(nan_out: *mut f32, padded_out: *mut u64) {
        let padded = get_padded_static();
        unsafe {
            *nan_out = *get_static_nan();
            *padded_out = ((padded.value as u64) << 8) | padded.tag as u64;
        }
    }

    #[kernel]
    pub unsafe fn static_subobject_pointers(out: *mut u32) {
        unsafe {
            *out.add(0) = *get_padded_static_tag() as u32;
            *out.add(1) = *get_padded_static_value();
            *out.add(2) = get_static_weight_pair()[0].to_bits();
            *out.add(3) = get_static_weight_pair()[1].to_bits();
        }
    }

    /// Read through pointer relocations stored inside device-global
    /// initializers. The table covers a direct target, repeated/shared targets,
    /// a second target, and an interior pointer with a non-zero byte addend.
    #[kernel]
    pub unsafe fn static_initializer_relocations(out: *mut u32) {
        unsafe {
            *out.add(0) = *RELOCATION_REFERENCE;
            *out.add(1) = *RELOCATION_REFERENCES[0];
            *out.add(2) = *RELOCATION_REFERENCES[1];
            *out.add(3) = (*INTERIOR_RELOCATION_REFERENCE).to_bits();
        }
    }

    /// A one-past-the-end constant pointer is formed and compared, never
    /// dereferenced. The distance from the static's base must equal the
    /// allocation size (32 bytes).
    #[kernel]
    pub unsafe fn static_one_past_end(out: *mut u32) {
        let base = get_static_weights() as *const [[f32; 2]; 4] as usize;
        let end = get_static_weights_end() as usize;
        unsafe {
            *out = (end - base) as u32;
        }
    }
}

fn main() {
    println!("=== Device Global Static Example ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let out_dev = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate output");

    let module = ctx
        .load_module_from_file("device_global.ptx")
        .expect("Failed to load PTX module");
    let module = kernels::from_module(module).expect("Failed to initialize typed CUDA module");

    for launch_idx in 1..=2 {
        unsafe {
            module.device_global(
                &stream,
                LaunchConfig::for_num_elems(1),
                out_dev.cu_deviceptr() as *mut u64,
            )
        }
        .expect("Kernel launch failed");

        let result = out_dev.to_host_vec(&stream).expect("Failed to copy result")[0];
        let expected = launch_idx ^ 0x00C0_FFEEu64;

        println!("Launch {launch_idx}: result = {result:#x}");
        if result != expected {
            eprintln!("FAILED: expected {expected:#x}, got {result:#x}");
            std::process::exit(1);
        }
    }

    let static_out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, 1).expect("Failed to allocate static output");
    unsafe {
        module.nonzero_static_table(
            &stream,
            LaunchConfig::for_num_elems(1),
            static_out_dev.cu_deviceptr() as *mut f32,
        )
    }
    .expect("Static table kernel launch failed");
    let static_result = static_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy static result")[0];
    let static_expected = 12.25f32;
    println!("Static table: result = {static_result}");
    if (static_result - static_expected).abs() > f32::EPSILON {
        eprintln!("FAILED: expected {static_expected}, got {static_result}");
        std::process::exit(1);
    }

    let nan_out_dev =
        DeviceBuffer::<f32>::zeroed(&stream, 1).expect("Failed to allocate NaN output");
    let padded_out_dev =
        DeviceBuffer::<u64>::zeroed(&stream, 1).expect("Failed to allocate padded output");
    unsafe {
        module.static_initializer_edges(
            &stream,
            LaunchConfig::for_num_elems(1),
            nan_out_dev.cu_deviceptr() as *mut f32,
            padded_out_dev.cu_deviceptr() as *mut u64,
        )
    }
    .expect("Static initializer edge-case kernel launch failed");

    let nan_bits = nan_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy NaN output")[0]
        .to_bits();
    let padded_result = padded_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy padded output")[0];
    let padded_expected = (0x1234_5678u64 << 8) | 0xabu64;
    println!("NaN payload: bits = {nan_bits:#010x}");
    println!("Padded static: result = {padded_result:#x}");
    if nan_bits != 0x7fc0_1234 || padded_result != padded_expected {
        eprintln!(
            "FAILED: expected NaN bits {:#010x} and padded value {padded_expected:#x}, got {nan_bits:#010x} and {padded_result:#x}",
            0x7fc0_1234u32
        );
        std::process::exit(1);
    }

    let subobject_out_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("Failed to allocate subobject output");

    unsafe {
        module.static_subobject_pointers(
            &stream,
            LaunchConfig::for_num_elems(1),
            subobject_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Static subobject kernel launch failed");

    let subobject_result = subobject_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy static subobject output");

    let subobject_expected = [0xabu32, 0x1234_5678, 4.0f32.to_bits(), 8.0f32.to_bits()];

    println!("Static subobjects: result = {subobject_result:?}");

    if subobject_result.as_slice() != subobject_expected.as_slice() {
        eprintln!(
            "FAILED: expected static subobjects {subobject_expected:?}, got {subobject_result:?}"
        );
        std::process::exit(1);
    }

    let relocation_out_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 4).expect("Failed to allocate relocation output");

    unsafe {
        module.static_initializer_relocations(
            &stream,
            LaunchConfig::for_num_elems(1),
            relocation_out_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("Static initializer relocation kernel launch failed");

    let relocation_result = relocation_out_dev
        .to_host_vec(&stream)
        .expect("Failed to copy relocation output");
    let relocation_expected = [0x1234_5678, 0x1234_5678, 0xcafe_babe, 8.0f32.to_bits()];

    println!("Static initializer relocations: result = {relocation_result:?}");

    if relocation_result.as_slice() != relocation_expected.as_slice() {
        eprintln!(
            "FAILED: expected static initializer relocations {relocation_expected:?}, got {relocation_result:?}"
        );
        std::process::exit(1);
    }

    let one_past_end_dev =
        DeviceBuffer::<u32>::zeroed(&stream, 1).expect("Failed to allocate one-past-end output");

    unsafe {
        module.static_one_past_end(
            &stream,
            LaunchConfig::for_num_elems(1),
            one_past_end_dev.cu_deviceptr() as *mut u32,
        )
    }
    .expect("One-past-end kernel launch failed");

    let one_past_end_result = one_past_end_dev
        .to_host_vec(&stream)
        .expect("Failed to copy one-past-end output")[0];

    println!("One-past-the-end offset: result = {one_past_end_result}");

    if one_past_end_result != 32 {
        eprintln!("FAILED: expected one-past-the-end offset 32, got {one_past_end_result}");
        std::process::exit(1);
    }

    println!(
        "\nSUCCESS: device globals preserved storage, initializer bytes, pointer relocations, pointer addends, and subobject addresses."
    );
}
