/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `CUDA_OXIDE_POST_IR` smoke test: an external hook edits the exported LLVM
//! IR and the pipeline still produces working PTX.
//!
//! Build and run with:
//!   cargo oxide run post_ir_hook
//!
//! `CUDA_OXIDE_POST_IR` names one or more executables (a PATH-style list) run
//! in order on the exported `.ll` between IR export and PTX generation. Each
//! hook is invoked as
//!
//!   <hook> <ll_path> <output_dir> <output_name> <target>
//!
//! and may rewrite `<ll_path>` in place; exit 0 continues the build on the
//! edited IR, non-zero aborts it with the hook's stderr. Hooks are
//! transform-only — stdout is ignored and the pipeline always finishes its
//! own PTX generation. (See the `enzyme_autodiff` example for a real
//! transform: automatic differentiation of a device function.)
//!
//! This example wires the hook up hermetically: `.cargo/config.toml` sets
//! `CUDA_OXIDE_POST_IR=hook.sh` (resolved relative to the example root) for
//! the device build, and `hook.sh` rewrites the `MARKER` constant below from
//! 1010101 to 2020202 in the exported IR. The kernel therefore computes
//! something its Rust source does not say — the GPU result proves the PTX was
//! generated from the hook-edited IR.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};

/// Baked into the kernel body; `hook.sh` rewrites it to [`HOOKED_MARKER`] in
/// the exported IR. Host code is compiled by the normal LLVM backend and never
/// passes through the hook, so this constant stays 1010101 here.
const MARKER: i32 = 1010101;
/// What the kernel actually adds after the hook has edited the IR.
const HOOKED_MARKER: i32 = 2020202;

#[cuda_module]
mod kernels {
    use super::*;

    /// `out[i] = x[i] + MARKER` — as written. As compiled, the post-IR hook
    /// has rewritten the constant, so the device computes `x[i] + 2020202`.
    #[kernel]
    pub fn add_marker(x: &[i32], mut out: DisjointSlice<i32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            *o = x[i].wrapping_add(MARKER);
        }
    }
}

fn main() {
    println!("=== CUDA_OXIDE_POST_IR smoke test ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let x_host: Vec<i32> = (0..N as i32).collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut out_dev = DeviceBuffer::<i32>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    // SAFETY: the launch is 1-D with one thread per element, matching
    // index_1d(); the buffers cover every access the kernel makes.
    unsafe {
        module.add_marker(
            &stream,
            LaunchConfig::for_num_elems(N as u32),
            &x_dev,
            &mut out_dev,
        )
    }
    .expect("Kernel launch failed");

    let out_host = out_dev.to_host_vec(&stream).unwrap();

    // The Rust source adds MARKER; the hook rewrote the IR to add
    // HOOKED_MARKER. Seeing HOOKED_MARKER on the GPU proves both halves of
    // the contract: the hook ran, and the pipeline generated working PTX from
    // the edited IR.
    let mut errors = 0;
    let mut hook_did_not_run = 0;
    for i in 0..N {
        let got = out_host[i];
        if got == x_host[i].wrapping_add(HOOKED_MARKER) {
            continue;
        }
        if got == x_host[i].wrapping_add(MARKER) {
            hook_did_not_run += 1;
        } else if errors < 5 {
            eprintln!(
                "  Error at [{}]: expected {}, got {}",
                i,
                x_host[i].wrapping_add(HOOKED_MARKER),
                got
            );
        }
        errors += 1;
    }

    if errors == 0 {
        println!(
            "✓ SUCCESS: all {} elements show the hook-rewritten constant ({} -> {})",
            N, MARKER, HOOKED_MARKER
        );
    } else if hook_did_not_run == errors {
        println!("✗ FAILED: kernel still adds the original MARKER — the post-IR hook did not run.");
        println!(
            "  (CUDA_OXIDE_POST_IR is set by this example's .cargo/config.toml; was it overridden?)"
        );
        std::process::exit(1);
    } else {
        println!("✗ FAILED: {} errors", errors);
        std::process::exit(1);
    }
}
