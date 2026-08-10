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

/// Folds the three results into one word per case, so the device can report
/// them through a `u64` slice and the host can compare without a layout
/// assumption.
fn case_results() -> [u64; 3] {
    let deref = through_deref(0) as u32 as u64;

    let field = through_field();
    let field = field.0.to_bits() ^ u64::from(field.1);

    let indexed = through_index(1);
    let index = (indexed[0] as u32 as u64)
        ^ ((indexed[1] as u32 as u64) << 8)
        ^ ((indexed[2] as u32 as u64) << 16);

    [deref, field, index]
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn projected_destinations(mut out: DisjointSlice<u64>) {
        let results = case_results();
        if let Some(slot) = out.get_mut(thread::index_1d()) {
            *slot = results[0] ^ results[1].rotate_left(21) ^ results[2].rotate_left(42);
        }
    }
}

fn main() {
    let host = case_results();
    let host_folded = host[0] ^ host[1].rotate_left(21) ^ host[2].rotate_left(42);

    println!("=== intrinsic calls writing through a projected destination ===\n");
    println!("host  deref case: 0x{:016x}", host[0]);
    println!("host  field case: 0x{:016x}", host[1]);
    println!("host  index case: 0x{:016x}", host[2]);

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
        println!("\nPASS: device and host agree on all three projections");
    } else {
        println!("\nFAIL: device and host disagree");
        std::process::exit(1);
    }
}
