/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Automatic differentiation of a device function with Enzyme, via the
//! `CUDA_OXIDE_POST_IR` hook.
//!
//! Build and run with:
//!   cargo oxide run enzyme_autodiff
//!
//! Requires an LLVMEnzyme plugin built against LLVM 21 (see README.md).
//! Without one, the hook leaves the IR untouched and this example prints
//! `skipping: ...` instead of failing.
//!
//! ## How it works
//!
//! `poly` is an ordinary `#[device]` function and `poly_dx` is a stub with the
//! same signature whose body just returns [`GRAD_SENTINEL`]. The `poly_grad`
//! kernel calls the stub per element. At build time, `enzyme.sh` (wired up by
//! `.cargo/config.toml`) edits the exported LLVM IR:
//!
//!   1. renames the stub's `define` out of the way, keeping call sites intact;
//!   2. appends a new `@<poly_dx>` definition whose body calls
//!      `__enzyme_fwddiff(@<poly>, x, 1.0)` — Enzyme's forward-mode request;
//!   3. runs `opt -load-pass-plugin=LLVMEnzyme… -passes='enzyme,default<O2>'`,
//!      which replaces that request with generated derivative code.
//!
//! The pipeline then finishes its normal PTX generation on the edited IR, and
//! the host below checks Enzyme's derivative against the analytic one. The
//! host build never passes through the hook, so calling `poly_dx` here on the
//! CPU would still return the sentinel — only the device code is rewired.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, cuda_module, device, kernel, thread};

/// What the `poly_dx` stub returns when Enzyme has NOT rewired it. The host
/// uses this to distinguish "hook skipped (no plugin)" from a wrong derivative.
const GRAD_SENTINEL: f64 = -9.876543210e98;

/// The primal: a polynomial so the example needs no libdevice math and the
/// derivative is hand-checkable.
///
///   f(x)  = x³ + 2x² − 5x + 1
///   f'(x) = 3x² + 4x − 5
#[device]
pub fn poly(x: f64) -> f64 {
    x * x * x + 2.0 * x * x - 5.0 * x + 1.0
}

/// Derivative stub. As written it returns [`GRAD_SENTINEL`]; as compiled (for
/// the device) the post-IR hook has replaced it with Enzyme's `d poly/dx`.
/// `#[inline(never)]` keeps the call in `poly_grad` from being inlined away
/// before the hook can retarget it.
#[device]
#[inline(never)]
pub fn poly_dx(x: f64) -> f64 {
    let _ = x;
    GRAD_SENTINEL
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Primal evaluation: `y[i] = poly(x[i])` — untouched by the hook.
    #[kernel]
    pub fn poly_eval(x: &[f64], mut y: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = y.get_mut(idx) {
            *o = poly(x[i]);
        }
    }

    /// Derivative evaluation: `dy[i] = poly_dx(x[i])`, where `poly_dx` is
    /// Enzyme-generated in the device build.
    #[kernel]
    pub fn poly_grad(x: &[f64], mut dy: DisjointSlice<f64>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = dy.get_mut(idx) {
            *o = poly_dx(x[i]);
        }
    }
}

/// The analytic derivative the Enzyme result must match.
fn poly_dx_analytic(x: f64) -> f64 {
    3.0 * x * x + 4.0 * x - 5.0
}

fn main() {
    println!("=== Enzyme autodiff via CUDA_OXIDE_POST_IR ===\n");

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let x_host: Vec<f64> = (0..N)
        .map(|i| -4.0 + 8.0 * (i as f64) / (N as f64))
        .collect();

    let x_dev = DeviceBuffer::from_host(&stream, &x_host).unwrap();
    let mut y_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();
    let mut dy_dev = DeviceBuffer::<f64>::zeroed(&stream, N).unwrap();

    let module = kernels::load(&ctx).expect("Failed to load embedded CUDA module");
    let cfg = LaunchConfig::for_num_elems(N as u32);
    // SAFETY: both launches are 1-D with one thread per element, matching
    // index_1d(); the buffers cover every access the kernels make.
    unsafe { module.poly_eval(&stream, cfg, &x_dev, &mut y_dev) }.expect("poly_eval launch failed");
    unsafe { module.poly_grad(&stream, cfg, &x_dev, &mut dy_dev) }
        .expect("poly_grad launch failed");

    let y_host = y_dev.to_host_vec(&stream).unwrap();
    let dy_host = dy_dev.to_host_vec(&stream).unwrap();

    // Hook skipped (no Enzyme plugin found): the stub still returns the
    // sentinel. Opt out gracefully so the example is runnable everywhere.
    if dy_host.iter().all(|&d| d == GRAD_SENTINEL) {
        println!("skipping: LLVMEnzyme plugin not available (enzyme.sh left the IR untouched)");
        println!("  build Enzyme against LLVM 21 and set LLVMENZYME=/path/to/LLVMEnzyme-21.so");
        return;
    }

    let mut primal_errors = 0;
    let mut grad_errors = 0;
    for i in 0..N {
        let x = x_host[i];
        if (y_host[i] - poly(x)).abs() > 1e-12 {
            if primal_errors < 3 {
                eprintln!(
                    "  primal error at x={x}: expected {}, got {}",
                    poly(x),
                    y_host[i]
                );
            }
            primal_errors += 1;
        }
        // O2 may reassociate the generated derivative, so compare with a
        // small relative tolerance rather than exactly.
        let want = poly_dx_analytic(x);
        if (dy_host[i] - want).abs() > 1e-9 * (1.0 + want.abs()) {
            if grad_errors < 3 {
                eprintln!(
                    "  gradient error at x={x}: expected {want}, got {}",
                    dy_host[i]
                );
            }
            grad_errors += 1;
        }
    }

    println!(
        "f(x)  = x^3 + 2x^2 - 5x + 1   checked at {} points: {}",
        N,
        ok(primal_errors)
    );
    println!(
        "f'(x) = 3x^2 + 4x - 5 (Enzyme) checked at {} points: {}",
        N,
        ok(grad_errors)
    );
    println!(
        "  e.g. f'({:.3}) = {:.6} (analytic {:.6})",
        x_host[N / 3],
        dy_host[N / 3],
        poly_dx_analytic(x_host[N / 3])
    );

    if primal_errors == 0 && grad_errors == 0 {
        println!("\n✓ SUCCESS: Enzyme's device derivative matches the analytic one");
    } else {
        println!(
            "\n✗ FAILED: {} primal / {} gradient mismatches",
            primal_errors, grad_errors
        );
        std::process::exit(1);
    }
}

fn ok(errors: usize) -> &'static str {
    if errors == 0 { "ok" } else { "MISMATCH" }
}
