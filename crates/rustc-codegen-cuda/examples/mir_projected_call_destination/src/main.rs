/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(custom_mir, core_intrinsics)]
#![allow(internal_features)]
// MIR-shaped bodies carry rustc's own local names and redundant temps, and
// custom MIR's `Call(..)` terminator reads to clippy as a unit argument.
#![allow(
    clippy::just_underscores_and_digits,
    clippy::similar_names,
    clippy::unit_arg
)]

//! An intrinsic call writing through a projected destination.
//!
//! rustc lowers an ordinary call whose destination carries a projection into a
//! call to a temporary followed by a store, so the projection never reaches
//! code generation. An intrinsic keeps its destination, which leaves three
//! shapes to translate: a dereferenced pointer, a struct field, and an array
//! element. Each is written here in custom MIR, since surface Rust cannot
//! produce them.
//!
//! Three translation paths build or store the call result themselves and so
//! have store sites of their own, exercised separately: a float-math
//! placeholder (`sqrtf32`), whose result must be typed from the projected
//! place rather than the whole local; a plain function call, whose result
//! the function-item path stores itself; and `libm::sincosf`, which packs a
//! `(sin, cos)` tuple and must write it through the projection. Those cases
//! are written so the shape actually reaches the importer under full
//! optimization: inlining rewrites a projected call destination into a
//! temporary plus a store, and GVN sees through a locally-taken pointer, so
//! the bodies are `inline(never)` and the sincos pointer arrives as an
//! opaque argument.
//!
//! Each case runs on the device and on the host from the same body, and the
//! two results must agree. A result that landed at the wrong address shows up
//! as a difference, since the host reads what the device wrote back.
//!
//! Build and run with:
//!   cargo oxide run mir_projected_call_destination

use core::intrinsics::mir::*;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

/// `(*p) = bswap(x)`, the pointer being to this function's own argument.
///
/// The result has to land in the pointee. Writing it to `_2` instead would
/// leave `_1` untouched, which the returned value reports.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_deref(mut _1: i32) -> i32 {
    mir! {
        type RET = i32;
        let _2: *mut i32;
        {
            _2 = core::ptr::addr_of_mut!(_1);
            Call((*_2) = core::intrinsics::bswap(451059808_i32), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET = _1;
            Return()
        }
    }
}

/// `RET.1 = bswap(x)` on a tuple whose other field is eight bytes wide.
///
/// The result has to land in the second field. Writing it to the whole tuple
/// asks for a cast from a byte to `{ double, i8, [7 x i8] }`, which is the
/// shape LLVM refuses.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_field() -> (f64, u8) {
    mir! {
        type RET = (f64, u8);
        {
            Call(RET.1 = core::intrinsics::bswap(7_u8), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 1.5_f64;
            Return()
        }
    }
}

/// `RET[i] = bswap(x)` with a runtime index.
///
/// The result has to land in the indexed element, leaving the other two at
/// the value the array was initialised with.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_index(mut _1: usize) -> [i32; 3] {
    mir! {
        type RET = [i32; 3];
        {
            RET = [11_i32, 22_i32, 33_i32];
            Call(RET[_1] = core::intrinsics::bswap(451059808_i32), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// `RET.1 = sqrtf32(x)` on a tuple whose other field is eight bytes wide.
///
/// Unlike `bswap`, a float-math intrinsic keeps a placeholder call in the
/// translation, and the placeholder's result is typed from the destination.
/// Typed from the whole tuple local instead of the projected field, the
/// store would aim a `{ i64, float }` at the field's `float` slot, which the
/// verifier refuses. `sqrt` is correctly rounded on host and device alike,
/// so the bit-exact comparison below is safe.
///
/// `inline(never)` matters: inlining rewrites a projected call destination
/// into a fresh temporary plus an ordinary store, which would erase the very
/// shape this case exists to reach.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_field_float(_1: f32) -> (u64, f32) {
    mir! {
        type RET = (u64, f32);
        {
            Call(RET.1 = core::intrinsics::sqrtf32(_1), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 3_u64;
            Return()
        }
    }
}

/// `RET[i] = sqrtf32(x)` with a runtime index, on a float array.
///
/// The index spelling of the same float-math shape. Unlike the field one it
/// survives even inlining (the inliner keeps index projections), so it
/// reaches the importer under every optimization decision.
#[custom_mir(dialect = "runtime", phase = "initial")]
fn through_index_float(mut _1: usize, _2: f32) -> [f32; 3] {
    mir! {
        type RET = [f32; 3];
        {
            RET = [1.0_f32, 1.0_f32, 1.0_f32];
            Call(RET[_1] = core::intrinsics::sqrtf32(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Callee for the plain-function case; `inline(never)` keeps the call (and
/// with it the projected destination) alive in the caller's MIR.
#[inline(never)]
fn double_it(x: u32) -> u32 {
    x.wrapping_mul(2)
}

/// `RET.1 = double_it(x)`: a plain function call, not an intrinsic.
///
/// Ordinary surface-Rust calls lower a projected destination to a temporary
/// before codegen, but custom MIR hands the projection straight to the
/// importer. The function-item path (and its closure sibling) has a store
/// site of its own, separate from the intrinsic ones, and must write the
/// call result through the projection there too.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_field_fn(_1: u32) -> (u64, u32) {
    mir! {
        type RET = (u64, u32);
        {
            Call(RET.1 = double_it(_1), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            RET.0 = 5_u64;
            Return()
        }
    }
}

/// `(*p) = sincosf(x)` through a pointer this function receives opaquely.
///
/// `sincos` packs the `(sin, cos)` pair itself before storing, so it has a
/// store site of its own: the pair is typed from the pointee and has to be
/// written through the pointer. Written to the pointer's local instead, a
/// tuple would be aimed at a pointer slot. The pointer must arrive as an
/// argument: were it materialized here from a local, GVN would see through
/// it and rewrite `(*p)` back to the plain local. The angle 0 keeps the
/// comparison exact: both sides produce `(+0.0, 1.0)` bitwise.
#[custom_mir(dialect = "runtime", phase = "initial")]
#[inline(never)]
fn through_deref_sincos(_1: *mut (f32, f32), _2: f32) {
    mir! {
        {
            Call((*_1) = libm::sincosf(_2), ReturnTo(bb1), UnwindUnreachable())
        }
        bb1 = {
            Return()
        }
    }
}

/// Folds the seven results into one word per case, so the device can report
/// them through a `u64` slice and the host can compare without a layout
/// assumption.
fn case_results() -> [u64; 7] {
    let deref = through_deref(0) as u32 as u64;

    let field = through_field();
    let field = field.0.to_bits() ^ u64::from(field.1);

    let indexed = through_index(1);
    let index = (indexed[0] as u32 as u64)
        ^ ((indexed[1] as u32 as u64) << 8)
        ^ ((indexed[2] as u32 as u64) << 16);

    let field_float = through_field_float(2.0_f32);
    let field_float = field_float.0 ^ u64::from(field_float.1.to_bits());

    let index_float = through_index_float(1, 2.0_f32);
    let index_float = u64::from(index_float[0].to_bits())
        ^ u64::from(index_float[1].to_bits()).rotate_left(8)
        ^ u64::from(index_float[2].to_bits()).rotate_left(16);

    let field_fn = through_field_fn(21);
    let field_fn = field_fn.0 ^ u64::from(field_fn.1);

    let mut pair = (9.0_f32, 9.0_f32);
    through_deref_sincos(&raw mut pair, 0.0_f32);
    let sincos = u64::from(pair.0.to_bits()) ^ (u64::from(pair.1.to_bits()) << 32);

    [
        deref,
        field,
        index,
        field_float,
        index_float,
        field_fn,
        sincos,
    ]
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn projected_destinations(mut out: DisjointSlice<u64>) {
        let results = case_results();
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = results[0]
                ^ results[1].rotate_left(8)
                ^ results[2].rotate_left(16)
                ^ results[3].rotate_left(24)
                ^ results[4].rotate_left(32)
                ^ results[5].rotate_left(40)
                ^ results[6].rotate_left(48);
        }
    }
}

fn main() {
    let host = case_results();
    let host_folded = host[0]
        ^ host[1].rotate_left(8)
        ^ host[2].rotate_left(16)
        ^ host[3].rotate_left(24)
        ^ host[4].rotate_left(32)
        ^ host[5].rotate_left(40)
        ^ host[6].rotate_left(48);

    println!("=== intrinsic calls writing through a projected destination ===\n");
    println!("host  deref case:       0x{:016x}", host[0]);
    println!("host  field case:       0x{:016x}", host[1]);
    println!("host  index case:       0x{:016x}", host[2]);
    println!("host  float field case: 0x{:016x}", host[3]);
    println!("host  float index case: 0x{:016x}", host[4]);
    println!("host  fn field case:    0x{:016x}", host[5]);
    println!("host  sincos case:      0x{:016x}", host[6]);

    let ctx = CudaContext::new(0).expect("failed to create CUDA context");
    let stream = ctx.default_stream();
    let mut out = DeviceBuffer::<u64>::zeroed(&stream, 1).expect("alloc out");
    let module = kernels::load(&ctx).expect("failed to load device module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    // SAFETY: the one argument matches `projected_destinations`' single slice
    // parameter, and `out` is a live DeviceBuffer allocated above.
    unsafe { module.projected_destinations(&stream, cfg, &mut out) }.expect("kernel launch failed");

    let device_folded = out.to_host_vec(&stream).expect("readback")[0];
    println!("\nhost  folded:     0x{host_folded:016x}");
    println!("device folded:    0x{device_folded:016x}");

    if device_folded == host_folded {
        println!("\nPASS: device and host agree on all seven projections");
    } else {
        println!("\nFAIL: device and host disagree");
        std::process::exit(1);
    }
}
