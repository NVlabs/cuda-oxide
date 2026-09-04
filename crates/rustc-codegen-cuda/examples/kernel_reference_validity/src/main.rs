/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! End-to-end LLVM-IR coverage for rustc-proven kernel reference validity.
//!
//! Build with `cargo oxide build kernel_reference_validity`, then run the host
//! binary (or `cargo oxide run kernel_reference_validity`) to inspect the
//! generated `.ll` parameter lines.

use cuda_device::{DisjointSlice, kernel, thread};

#[repr(align(16))]
pub struct AlignedZst;

#[derive(Clone, Copy)]
pub struct ByValue {
    pub pointer: *const f32,
}

#[kernel]
pub fn shared_ref(_value: &f32) {}

#[kernel]
pub fn unique_ref(_value: &mut f32) {}

#[kernel]
pub fn shared_slice(_value: &[f32]) {}

#[kernel]
pub fn unique_slice(_value: &mut [f32]) {}

#[kernel]
pub fn align_one(_value: &u8) {}

#[kernel]
pub fn aligned_zst(_value: &AlignedZst) {}

#[kernel]
pub fn raw_pointer(_value: *const f32) {}

#[kernel]
pub fn disjoint_slice(_value: DisjointSlice<f32>) {}

/// Forces the module through the libdevice/libNVVM finalization path while
/// keeping one audited slice reference and one bare `DisjointSlice` control in
/// the same kernel signature.
#[kernel]
pub fn nvvm_slice(input: &[f32], mut output: DisjointSlice<f32>) {
    let index = thread::index_1d();
    let i = index.get();
    if let Some(out) = output.get_mut(index) {
        *out = input[i].exp();
    }
}

#[kernel]
pub fn by_value(_value: ByValue) {}

#[allow(improper_ctypes_definitions)]
#[kernel]
pub extern "C" fn c_reference(_value: &f32) {}

fn kernel_header<'a>(llvm_ir: &'a str, name: &str) -> Result<&'a str, Box<dyn std::error::Error>> {
    let needle = format!("@{name}(");
    llvm_ir
        .lines()
        .find(|line| line.trim_start().starts_with("define ") && line.contains(&needle))
        .ok_or_else(|| format!("missing LLVM kernel definition `{name}`").into())
}

fn require_reference_attrs(
    llvm_ir: &str,
    name: &str,
    alignment: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    let expected = format!("nonnull align {alignment}");
    if !header.contains(&expected) {
        return Err(format!(
            "kernel `{name}` is missing `{expected}` on its reference parameter:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_nonnull_without_alignment(
    llvm_ir: &str,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if !header.contains("nonnull") || header.contains(" align ") {
        return Err(format!(
            "kernel `{name}` must carry nonnull without a redundant align-1 attribute:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_single_slice_pointer_fact(
    llvm_ir: &str,
    name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if header.matches("nonnull").count() != 1 || header.matches("align 4").count() != 1 {
        return Err(format!(
            "kernel `{name}` must annotate only the slice data pointer:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn require_bare(llvm_ir: &str, name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let header = kernel_header(llvm_ir, name)?;
    if header.contains("nonnull") || header.contains(" align ") {
        return Err(format!(
            "kernel `{name}` unexpectedly acquired Rust-reference validity:\n{header}"
        )
        .into());
    }
    Ok(())
}

fn verify_generated_llvm_ir() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("kernel_reference_validity.ll");
    let llvm_ir = std::fs::read_to_string(&path)?;

    require_reference_attrs(&llvm_ir, "shared_ref", 4)?;
    require_reference_attrs(&llvm_ir, "unique_ref", 4)?;
    require_single_slice_pointer_fact(&llvm_ir, "shared_slice")?;
    require_single_slice_pointer_fact(&llvm_ir, "unique_slice")?;
    require_single_slice_pointer_fact(&llvm_ir, "nvvm_slice")?;
    require_nonnull_without_alignment(&llvm_ir, "align_one")?;
    require_reference_attrs(&llvm_ir, "aligned_zst", 16)?;

    require_bare(&llvm_ir, "raw_pointer")?;
    require_bare(&llvm_ir, "disjoint_slice")?;
    require_bare(&llvm_ir, "by_value")?;
    require_bare(&llvm_ir, "c_reference")?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    verify_generated_llvm_ir()?;
    println!("kernel_reference_validity: PASS (nonnull/align kernel parameter policy)");
    Ok(())
}
