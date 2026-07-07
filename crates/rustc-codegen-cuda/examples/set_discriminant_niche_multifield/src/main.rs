/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Positive test: `SetDiscriminant` on a niche-encoded enum whose data variant
//! has SEVERAL fields, where the niche lives in a LATER field.
//!
//! `enum Multi { Nothing, Something(u32, NonZeroU32) }` is niche-encoded: the
//! `NonZeroU32` (the SECOND field of `Something`) provides the niche, so rustc
//! stores `Nothing` as `Something` with that field set to `0`. There is no
//! separate tag. Setting the niche variant must therefore write the niche bit
//! pattern into the *second* payload field, not the first `u32`.
//!
//! What this exercises that the single-field example does not:
//!   * the importer's `niche_field_location` on a real MULTI-field rustc layout
//!     (it must pick field index 1, not 0), and
//!   * the offset-based niche-scalar locator in lowering producing loadable PTX
//!     for that field.
//!
//! Note on scope: the device determines the live variant from cuda-oxide's
//! synthetic discriminant tag, and a niche field value (`NonZeroU32 == 0`) is an
//! invalid value that cannot be read back without UB, so the *field-precise*
//! correctness (niche lands in slot 1, not slot 0) is pinned by the mir-lower
//! unit test `convert_set_discriminant_niche_targets_correct_multifield_slot`.
//! This example verifies the multi-field niche enum lowers, loads, and runs
//! correctly end-to-end on the device.
//!
//! Usage:
//!   cargo oxide run set_discriminant_niche_multifield

#![feature(core_intrinsics, custom_mir)]
#![allow(internal_features)]

use core::intrinsics::mir::*;
use core::num::NonZeroU32;
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Niche-encoded enum whose data variant carries two fields; the niche is in
/// the second one (`NonZeroU32`). The payload fields are written but never read
/// back (reading a niche field after `SetDiscriminant` would be UB), so the
/// enum is `dead_code`-allowed.
#[allow(dead_code)]
enum Multi {
    Nothing,
    Something(u32, NonZeroU32),
}

#[custom_mir(dialect = "runtime", phase = "optimized")]
fn force_set_nothing(e: &mut Multi) {
    mir!({
        // Variant index 0 == `Nothing`, the niche variant.
        SetDiscriminant(*e, 0);
        Return()
    })
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Each thread builds `Something(7, NonZeroU32::new(42))` (the untagged data
    /// variant), confirms it reads back as `Something`, then uses custom MIR to
    /// `SetDiscriminant` to `Nothing` (variant index 0). Output is `1` when the
    /// niche write flips the observed variant to `Nothing`, `0` otherwise.
    #[kernel]
    pub fn set_discriminant_niche_multifield_kernel(mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            let nz = unsafe { NonZeroU32::new_unchecked(42) };
            let mut e: Multi = Multi::Something(7, nz);

            // Sanity: the freshly built value is the data variant. `..` avoids
            // binding (and thus validating) the niche field.
            let was_something = matches!(e, Multi::Something(..));

            // Emits `StatementKind::SetDiscriminant(*e, 0)` directly.
            force_set_nothing(&mut e);

            // Reads cuda-oxide's synthetic discriminant tag. The `Something(..)`
            // arm never inspects the fields, so no niche-field validity is
            // required.
            let now_nothing = matches!(e, Multi::Nothing);

            *out_elem = if was_something && now_nothing { 1 } else { 0 };
        }
    }
}

fn main() {
    println!("=== set_discriminant_niche_multifield ===");
    println!(
        "Verifying multi-field niche SetDiscriminant (niche in the 2nd payload field) lowers, loads, and runs on the device."
    );

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 64;
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    unsafe {
        module
            .set_discriminant_niche_multifield_kernel(
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
                eprintln!("  Error at [{}]: expected 1 (Nothing), got {}", i, v);
            }
        }
    }

    if errors == 0 {
        println!(
            "PASS: all {} threads observed the multi-field niche SetDiscriminant write.",
            N
        );
    } else {
        eprintln!(
            "FAIL: {} threads did not observe the multi-field niche SetDiscriminant write.",
            errors
        );
        std::process::exit(1);
    }
}
