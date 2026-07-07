/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive test: niche-encoded enum discriminant writes via `SetDiscriminant`
//! are lowered correctly on the device.
//!
//! `Option<NonZeroU32>` is niche-encoded as a single `u32` where `0` means
//! `None`. The custom MIR helper emits `StatementKind::SetDiscriminant` to the
//! niche variant (`None`, variant index 0) so lowering must write both the
//! synthetic discriminant and the payload niche value.
//!
//! Usage:
//!   cargo oxide run set_discriminant_niche

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use core::num::NonZeroU32;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_set_niche_none(opt: &mut Option<NonZeroU32>) {
    mir!({
        SetDiscriminant(*opt, 0);
        Return()
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread starts with `Some(NonZeroU32::new(42))`, then uses custom
    /// MIR to emit `SetDiscriminant` to `None` (variant index 0). The output is
    /// `1` if the discriminant write was observed, `0` otherwise.
    #[kernel]
    pub fn set_discriminant_niche_kernel(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let some_value = unsafe { NonZeroU32::new_unchecked(42) };
            let mut opt: Option<NonZeroU32> = Some(some_value);

            // This helper emits `StatementKind::SetDiscriminant` directly.
            force_set_niche_none(&mut opt);

            *out_elem = match opt {
                Some(_) => 0,
                None => 1,
            };
        }
    }
}

fn main() {
    println!("=== set_discriminant_niche ===");
    println!(
        "Verifying that MIR SetDiscriminant lowers to a device-side tag + payload write for niche-encoded enums."
    );

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module
            .set_discriminant_niche_kernel(
                &stream,
                LaunchConfig::for_num_elems(N as u32),
                &mut out_dev,
            )
            .expect("Kernel launch failed");
    }

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for (i, &v) in out_host.iter().enumerate() {
        if v != 1 {
            errors += 1;
            if errors <= 5 {
                eprintln!("  Error at [{}]: expected 1 (None), got {}", i, v);
            }
        }
    }

    if errors == 0 {
        println!(
            "PASS: all {} threads observed the niche SetDiscriminant write.",
            N
        );
    } else {
        eprintln!(
            "FAIL: {} threads did not observe the niche SetDiscriminant write.",
            errors
        );
        std::process::exit(1);
    }
}
