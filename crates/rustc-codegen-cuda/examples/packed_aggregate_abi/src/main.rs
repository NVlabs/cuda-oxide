/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end ABI regression coverage for packed aggregates.
//!
//! This example exercises four paths that must agree on the same rustc byte
//! layout:
//!
//! - packed structs passed by value across the host -> kernel boundary;
//! - packed structs passed to and returned from an internal device helper;
//! - whole-value stores of packed structs to device memory;
//! - whole-value loads of packed structs from device memory.
//!
//! `Packed1` has no interior padding and occupies 5 bytes. `Packed2` has one
//! explicit padding byte between `a` and `b`, so it occupies 6 bytes and places
//! `b` at byte offset 2. The padding byte itself is intentionally not checked:
//! Rust does not guarantee a stable value for padding bytes.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct Packed1 {
    pub a: u8,
    pub b: u32,
}

#[repr(C, packed(2))]
#[derive(Clone, Copy)]
pub struct Packed2 {
    pub a: u8,
    pub b: u32,
}

#[cuda_module]
mod kernels {
    use super::*;

    #[inline(never)]
    fn transform_packed1(value: Packed1) -> Packed1 {
        let a = value.a;
        let b = value.b;
        Packed1 {
            a: a.wrapping_add(1),
            b: b.wrapping_add(0x0102_0304),
        }
    }

    #[inline(never)]
    fn transform_packed2(value: Packed2) -> Packed2 {
        let a = value.a;
        let b = value.b;
        Packed2 {
            a: a.wrapping_add(2),
            b: b.wrapping_add(0x1112_1314),
        }
    }

    #[kernel]
    pub unsafe fn packed1(value: Packed1, out: *mut u32) {
        let value = transform_packed1(value);
        let a = value.a;
        let b = value.b;

        // SAFETY: the host provides two writable u32 elements in `out`.
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn packed2(value: Packed2, out: *mut u32) {
        let value = transform_packed2(value);
        let a = value.a;
        let b = value.b;

        // SAFETY: the host provides two writable u32 elements in `out`.
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn store_packed1(value: Packed1, dst: *mut Packed1) {
        // SAFETY: `dst` points to device storage large enough for one Packed1.
        unsafe { dst.write(value) };
    }

    #[kernel]
    pub unsafe fn load_packed1(src: *const Packed1, out: *mut u32) {
        // SAFETY: `src` points to one initialized Packed1 and `out` has two u32s.
        let value = unsafe { src.read() };
        let a = value.a;
        let b = value.b;
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
    }

    #[kernel]
    pub unsafe fn store_packed2(value: Packed2, dst: *mut Packed2) {
        // SAFETY: `dst` points to device storage large enough and sufficiently
        // aligned for one Packed2. CUDA allocations satisfy this alignment.
        unsafe { dst.write(value) };
    }

    #[kernel]
    pub unsafe fn load_packed2(src: *const Packed2, out: *mut u32) {
        // SAFETY: `src` points to one initialized Packed2 and `out` has two u32s.
        let value = unsafe { src.read() };
        let a = value.a;
        let b = value.b;
        unsafe {
            out.write(u32::from(a));
            out.add(1).write(b);
        }
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

fn require_aggregate_parameter(
    ptx: &str,
    name: &str,
    byte_size: usize,
    abi_align: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = entry_header(ptx, name)?;
    let size_token = format!("[{byte_size}]");
    let align_token = format!(".param .align {abi_align} .b8");

    if !header.contains(&size_token) || !header.contains(&align_token) {
        return Err(format!(
            "kernel `{name}` does not expose the expected {byte_size}-byte, align-{abi_align} aggregate parameter:\n{header}"
        )
        .into());
    }

    Ok(())
}

fn verify_generated_ptx() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("packed_aggregate_abi.ptx");
    let ptx = std::fs::read_to_string(&path)?;

    require_aggregate_parameter(
        &ptx,
        "packed1",
        core::mem::size_of::<Packed1>(),
        core::mem::align_of::<Packed1>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "packed2",
        core::mem::size_of::<Packed2>(),
        core::mem::align_of::<Packed2>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "store_packed1",
        core::mem::size_of::<Packed1>(),
        core::mem::align_of::<Packed1>(),
    )?;
    require_aggregate_parameter(
        &ptx,
        "store_packed2",
        core::mem::size_of::<Packed2>(),
        core::mem::align_of::<Packed2>(),
    )?;

    Ok(())
}

fn assert_host_layout() {
    assert_eq!(core::mem::size_of::<Packed1>(), 5);
    assert_eq!(core::mem::align_of::<Packed1>(), 1);
    assert_eq!(core::mem::offset_of!(Packed1, a), 0);
    assert_eq!(core::mem::offset_of!(Packed1, b), 1);

    assert_eq!(core::mem::size_of::<Packed2>(), 6);
    assert_eq!(core::mem::align_of::<Packed2>(), 2);
    assert_eq!(core::mem::offset_of!(Packed2, a), 0);
    assert_eq!(core::mem::offset_of!(Packed2, b), 2);
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    assert_host_layout();

    if std::env::args().any(|arg| arg == "--verify-ptx") {
        verify_generated_ptx()?;
        println!("packed_aggregate_abi: PASS (host layout and PTX parameter shapes)");
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

    let by_value1_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let by_value2_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let load1_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let load2_out = DeviceBuffer::<u32>::zeroed(&stream, 2)?;
    let storage1 = DeviceBuffer::<u8>::zeroed(&stream, core::mem::size_of::<Packed1>())?;
    let storage2 = DeviceBuffer::<u8>::zeroed(&stream, core::mem::size_of::<Packed2>())?;

    let input1 = Packed1 {
        a: 0x21,
        b: 0x1020_3040,
    };
    let input2 = Packed2 {
        a: 0x31,
        b: 0x5060_7080,
    };

    let stored1 = Packed1 {
        a: 0x41,
        b: 0x90a0_b0c0,
    };
    let stored2 = Packed2 {
        a: 0x51,
        b: 0xd0e0_f001,
    };

    // SAFETY: every kernel launches one thread. The u32 output buffers contain
    // two writable elements. The byte buffers are CUDA allocations, so their
    // base addresses satisfy Packed1/Packed2 alignment and have exact storage
    // for one value of the corresponding type.
    unsafe {
        module.packed1(
            &stream,
            config,
            input1,
            by_value1_out.cu_deviceptr() as *mut u32,
        )?;
        module.packed2(
            &stream,
            config,
            input2,
            by_value2_out.cu_deviceptr() as *mut u32,
        )?;

        module.store_packed1(
            &stream,
            config,
            stored1,
            storage1.cu_deviceptr() as *mut Packed1,
        )?;
        module.load_packed1(
            &stream,
            config,
            storage1.cu_deviceptr() as *const Packed1,
            load1_out.cu_deviceptr() as *mut u32,
        )?;

        module.store_packed2(
            &stream,
            config,
            stored2,
            storage2.cu_deviceptr() as *mut Packed2,
        )?;
        module.load_packed2(
            &stream,
            config,
            storage2.cu_deviceptr() as *const Packed2,
            load2_out.cu_deviceptr() as *mut u32,
        )?;
    }

    assert_eq!(
        by_value1_out.to_host_vec(&stream)?,
        [0x22, 0x1122_3344]
    );
    assert_eq!(
        by_value2_out.to_host_vec(&stream)?,
        [0x33, 0x6172_8394]
    );
    assert_eq!(load1_out.to_host_vec(&stream)?, [0x41, 0x90a0_b0c0]);
    assert_eq!(load2_out.to_host_vec(&stream)?, [0x51, 0xd0e0_f001]);

    let bytes1 = storage1.to_host_vec(&stream)?;
    assert_eq!(bytes1[0], 0x41);
    assert_eq!(&bytes1[1..5], &0x90a0_b0c0u32.to_le_bytes());

    let bytes2 = storage2.to_host_vec(&stream)?;
    assert_eq!(bytes2[0], 0x51);
    // bytes2[1] is Rust padding and intentionally has no asserted value.
    assert_eq!(&bytes2[2..6], &0xd0e0_f001u32.to_le_bytes());

    println!(
        "packed_aggregate_abi: PASS (runtime values, whole-value load/store, and PTX parameter shapes)"
    );
    Ok(())
}
