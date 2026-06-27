/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Adversarial typed-pointer coverage for the legacy LLVM 7 NVVM dialect.
//!
//! cuda-oxide intentionally uses opaque pointers in its normal internal LLVM
//! representation. Legacy NVVM output canonicalizes erased pointer values to
//! byte pointers, then casts them at typed uses. This example makes the
//! pointer flows that defeated a local load/store-only repair strategy survive
//! into codegen: control-flow merges, a loop backedge, calls and returns,
//! nested aggregate fields, pointer-to-pointer memory, and two concrete views
//! of the same address.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[derive(Copy, Clone)]
struct PointerPair {
    first: *const f32,
    second: *const f32,
}

#[derive(Copy, Clone)]
struct PointerNest {
    pair: PointerPair,
    fallback: *const f32,
}

/// The returned pointer is selected by control flow. With inlining disabled,
/// the call boundary and the pointer merge both remain visible to codegen.
#[inline(never)]
fn choose_nested(nested: &PointerNest, selector: u32) -> *const f32 {
    if selector == 0 {
        nested.pair.first
    } else if selector == 1 {
        nested.pair.second
    } else {
        nested.fallback
    }
}

/// Carry a pointer around a runtime loop, producing a pointer-valued loop PHI
/// (represented as a block argument before textual LLVM export).
#[inline(never)]
unsafe fn walk_pointer(mut ptr: *const f32, mut steps: usize) -> *const f32 {
    while steps != 0 {
        ptr = unsafe { ptr.add(1) };
        steps -= 1;
    }
    ptr
}

/// A separate conditional control-flow merge exercises another pointer-valued
/// call/return path after the nested aggregate has been read.
#[inline(never)]
fn select_pointer(first: *const f32, second: *const f32, use_second: bool) -> *const f32 {
    if use_second { second } else { first }
}

/// Loading through `*const *const f32` keeps the pointer array in memory and
/// requires the legacy exporter to type both the slot address and its value.
#[inline(never)]
unsafe fn load_pointer_slot(slot: *const *const f32) -> *const f32 {
    unsafe { *slot }
}

/// The same erased address is intentionally consumed as both `float*` and
/// `i32*`-equivalent views. There is no single pointee type to infer globally;
/// each use supplies the type it needs.
#[inline(never)]
unsafe fn read_two_views(ptr: *const f32) -> (f32, u32) {
    let value = unsafe { *ptr };
    let bits = unsafe { *(ptr.cast::<u32>()) };
    (value, bits)
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn pointer_shapes(values: &[f32], mut out: DisjointSlice<[u64; 2]>) {
        let idx = thread::index_1d();
        let i = idx.get();
        let len = values.len();

        if i < len
            && let Some(row) = out.get_mut(idx)
        {
            let base = values.as_ptr();

            // All derived offsets remain in-bounds. The modulo operations are
            // runtime-dependent, preventing constant folding of the choices.
            let nested = PointerNest {
                pair: PointerPair {
                    first: unsafe { base.add((i + 1) % len) },
                    second: unsafe { base.add((i + 3) % len) },
                },
                fallback: unsafe { base.add((i + 5) % len) },
            };

            let from_nested = choose_nested(&nested, (i % 3) as u32);
            let from_loop = unsafe { walk_pointer(base, i) };
            let selected = select_pointer(from_nested, from_loop, i & 1 != 0);

            // Passing an element address to an opaque helper prevents the
            // pointer array from being reduced to a scalar select in MIR.
            let pointer_slots = [
                selected,
                nested.pair.first,
                nested.pair.second,
                nested.fallback,
            ];
            let slot = (i >> 1) & 3;
            let final_ptr = unsafe { load_pointer_slot(pointer_slots.as_ptr().add(slot)) };
            let (value, bits) = unsafe { read_two_views(final_ptr) };

            // `exp` forces the libdevice/NVVM-IR path even without an explicit
            // `--emit-nvvm-ir` flag. Store bits so the host can compare both
            // the math result and the differently typed pointer read exactly.
            row[0] = value.exp().to_bits() as u64;
            row[1] = bits as u64;
        }
    }
}

fn selected_offset(i: usize, len: usize) -> usize {
    let first = (i + 1) % len;
    let second = (i + 3) % len;
    let fallback = (i + 5) % len;
    let nested = match i % 3 {
        0 => first,
        1 => second,
        _ => fallback,
    };
    let selected = if i & 1 != 0 { i } else { nested };

    match (i >> 1) & 3 {
        0 => selected,
        1 => first,
        2 => second,
        _ => fallback,
    }
}

fn ulp_distance(a: f32, b: f32) -> u32 {
    let ordered = |value: f32| {
        let bits = value.to_bits();
        if bits & 0x8000_0000 != 0 {
            0x8000_0000 - (bits & 0x7fff_ffff)
        } else {
            0x8000_0000 + bits
        }
    };
    ordered(a).abs_diff(ordered(b))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 32;
    const EXP_ULP_LIMIT: u32 = 2;

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    // Exactly representable binary fractions keep the raw-bit assertion
    // independent of host/device arithmetic differences.
    let values: Vec<f32> = (0..N).map(|i| 0.25 + i as f32 / 32.0).collect();
    let values_dev = DeviceBuffer::from_host(&stream, &values)?;
    let mut out_dev = DeviceBuffer::<[u64; 2]>::zeroed(&stream, N)?;

    module.pointer_shapes(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &values_dev,
        &mut out_dev,
    )?;

    let got = out_dev.to_host_vec(&stream)?;
    let mut failures = 0usize;

    for i in 0..N {
        let source = values[selected_offset(i, N)];
        let got_exp = f32::from_bits(got[i][0] as u32);
        let expected_exp = source.exp();
        let exp_ulp = ulp_distance(got_exp, expected_exp);
        let got_source_bits = got[i][1] as u32;

        if exp_ulp > EXP_ULP_LIMIT || got_source_bits != source.to_bits() {
            failures += 1;
            if failures <= 8 {
                eprintln!(
                    "lane {i}: source={source}, exp GPU={got_exp:e}, CPU={expected_exp:e}, \
                     ULP={exp_ulp}; source bits GPU={got_source_bits:#010x}, \
                     CPU={:#010x}",
                    source.to_bits()
                );
            }
        }
    }

    if failures != 0 {
        return Err(format!("{failures}/{N} pointer-shape lanes failed").into());
    }

    println!(
        "PASS: {N} lanes across pointer control-flow merges, loop backedges, calls, nested aggregates, and pointer-to-pointer memory"
    );
    Ok(())
}
