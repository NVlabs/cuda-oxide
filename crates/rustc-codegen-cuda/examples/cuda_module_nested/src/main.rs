/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Nested-module `#[cuda_module]` example.
//!
//! Kernels live at three levels of nesting, one per collection mechanism:
//!
//! - `fill_index` at the module root (the pre-existing flat layout),
//! - `scale::scale_by` in an inline nested `mod`,
//! - `offset::offset_by` spliced in with `include!("stages/offset.rs")`,
//! - `double::double_all` in an out-of-line `mod double;`
//!   (`src/kernels/double.rs`).
//!
//! The generated launcher API stays flat: every kernel becomes a method on
//! `kernels::LoadedModule` regardless of nesting depth, so kernel names must
//! be unique across the whole module tree.
//!
//! Build and run with:
//!   cargo oxide run cuda_module_nested

// Required only for the out-of-line `pub mod double;` below: rustc gates
// non-inline modules in proc-macro input (rust-lang/rust#54727). Inline
// nested modules and include! need no feature gate.
#![feature(proc_macro_hygiene)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

#[cuda_module]
mod kernels {
    use super::*;

    /// Root-level kernel: out[i] = i
    #[kernel]
    pub fn fill_index(mut out: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let idx_raw = idx.get();
        if let Some(elem) = out.get_mut(idx) {
            *elem = idx_raw as f32;
        }
    }

    /// Inline nested module: out[i] = a[i] * 2
    pub mod scale {
        use cuda_device::{DisjointSlice, kernel, thread};

        #[kernel]
        pub fn scale_by(a: &[f32], mut out: DisjointSlice<f32>) {
            let idx = thread::index_1d();
            let idx_raw = idx.get();
            if let Some(elem) = out.get_mut(idx) {
                *elem = a[idx_raw] * 2.0;
            }
        }
    }

    /// include!-backed nested module: out[i] = a[i] + 10
    /// (rustc and the macro both resolve the literal relative to this file.)
    pub mod offset {
        include!("stages/offset.rs");
    }

    /// Out-of-line nested module: out[i] = a[i] + a[i]
    /// (resolved by module directory: src/kernels/double.rs)
    pub mod double;
}

fn main() {
    println!("=== #[cuda_module] Nested Modules Test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let mut idx_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut scaled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut offset_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();
    let mut doubled_dev = DeviceBuffer::<f32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let config = LaunchConfig::for_num_elems(N as u32);

    // Root kernel feeds the three nested kernels.
    module
        .fill_index(&stream, config, &mut idx_dev)
        .expect("fill_index launch failed");
    module
        .scale_by(&stream, config, &idx_dev, &mut scaled_dev)
        .expect("scale_by launch failed");
    module
        .offset_by(&stream, config, &idx_dev, &mut offset_dev)
        .expect("offset_by launch failed");
    module
        .double_all(&stream, config, &idx_dev, &mut doubled_dev)
        .expect("double_all launch failed");

    let scaled = scaled_dev.to_host_vec(&stream).unwrap();
    let offset = offset_dev.to_host_vec(&stream).unwrap();
    let doubled = doubled_dev.to_host_vec(&stream).unwrap();

    let mut errors = 0;
    for i in 0..N {
        let expected_scaled = i as f32 * 2.0;
        let expected_offset = i as f32 + 10.0;
        let expected_doubled = i as f32 + i as f32;
        if (scaled[i] - expected_scaled).abs() > 1e-5
            || (offset[i] - expected_offset).abs() > 1e-5
            || (doubled[i] - expected_doubled).abs() > 1e-5
        {
            if errors < 5 {
                eprintln!(
                    "  Error at [{i}]: scaled {} (want {expected_scaled}), offset {} (want {expected_offset}), doubled {} (want {expected_doubled})",
                    scaled[i], offset[i], doubled[i],
                );
            }
            errors += 1;
        }
    }

    if errors == 0 {
        println!("✓ SUCCESS: root, inline-mod, include!, and out-of-line kernels all ran");
    } else {
        println!("✗ FAILED: {errors} errors");
        std::process::exit(1);
    }
}
