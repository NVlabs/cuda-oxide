/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Hand-written `__nv_*` libdevice externs called straight from a kernel.
//!
//! Regression test for two declaration paths that must coexist:
//!
//! - A plain `extern "C" { fn __nv_asinf(...) }` block (no `#[device]`).
//!   The foreign item has no MIR body, so the importer must emit the call
//!   under the link symbol and mir-lower must declare it at the call site.
//!   Before this worked, the call failed module verification with
//!   "Symbol __nv_asinf not found".
//! - The `#[device] extern "C"` form of the same shape (`__nv_acosf`).
//!   The pipeline's device-extern declaration step also declares this
//!   symbol, so it must skip symbols the call-site path already declared
//!   instead of producing a multiple-definition error.
//!
//! Both symbols resolve against `libdevice.10.bc` when the NVVM IR is
//! linked via libNVVM + nvJitLink (same flow as `math_atan`).
//!
//! Run:
//!     cargo oxide run extern_libdevice
//!
//! Exits 0 on SUCCESS, 1 on FAIL.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, device, kernel, thread};
use cuda_host::{cuda_module, ltoir};

// Plain extern block: the original motivating shape. No macro support;
// the kernel calls libdevice directly.
unsafe extern "C" {
    fn __nv_asinf(x: f32) -> f32;
}

// #[device] extern route for a libdevice symbol: also declared by the
// pipeline's device-extern step, exercising declaration idempotence.
#[device]
unsafe extern "C" {
    fn __nv_acosf(x: f32) -> f32;
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn asin_acos(
        xs: &[f32],
        mut out_asin: DisjointSlice<f32>,
        mut out_acos: DisjointSlice<f32>,
    ) {
        let i = thread::index_1d().get();
        if i < xs.len() {
            let x = xs[i];
            // `ThreadIndex` is not `Copy`; mint one per write surface.
            if let Some(slot) = out_asin.get_mut(thread::index_1d()) {
                *slot = unsafe { __nv_asinf(x) };
            }
            if let Some(slot) = out_acos.get_mut(thread::index_1d()) {
                *slot = unsafe { __nv_acosf(x) };
            }
        }
    }
}

/// IEEE-754 ULP distance for finite f32 operands. `asin`/`acos` of inputs
/// in `[-1, 1]` land in `[-pi/2, pi]`, so NaN/Inf handling is not needed.
fn ulp_diff_f32(a: f32, b: f32) -> u64 {
    const SIGN: u64 = 0x8000_0000;
    const BODY: u64 = 0x7FFF_FFFF;
    let map = |bits: u64| {
        if bits & SIGN != 0 {
            SIGN - (bits & BODY)
        } else {
            SIGN + (bits & BODY)
        }
    };
    map(a.to_bits() as u64).abs_diff(map(b.to_bits() as u64))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== extern_libdevice: direct __nv_asinf/__nv_acosf calls ===\n");

    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    // `__nv_*` calls force the NVVM-IR output flavor; the first launch
    // builds a cubin via libNVVM + nvJitLink (links libdevice.10.bc).
    let module = ltoir::load_kernel_module(&ctx, "extern_libdevice")?;
    let module = kernels::from_module(module)?;

    // Full asin/acos domain including both endpoints.
    let xs: Vec<f32> = vec![
        -1.0, -0.9, -0.75, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0,
    ];
    let n = xs.len();
    let cfg = LaunchConfig::for_num_elems(n as u32);

    let xs_dev = DeviceBuffer::from_host(&stream, &xs)?;
    let mut out_asin = DeviceBuffer::<f32>::zeroed(&stream, n)?;
    let mut out_acos = DeviceBuffer::<f32>::zeroed(&stream, n)?;

    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.asin_acos(&stream, cfg, &xs_dev, &mut out_asin, &mut out_acos) }?;

    let got_asin = out_asin.to_host_vec(&stream)?;
    let got_acos = out_acos.to_host_vec(&stream)?;

    // libdevice transcendentals are typically within 1 ULP of host libm;
    // 2 ULP matches the bound math_atan / primitive_stress use.
    const ULP_LIMIT: u64 = 2;

    let mut failures = 0usize;
    for i in 0..n {
        let exp_asin = xs[i].asin();
        let exp_acos = xs[i].acos();
        let d_asin = ulp_diff_f32(got_asin[i], exp_asin);
        let d_acos = ulp_diff_f32(got_acos[i], exp_acos);
        if d_asin > ULP_LIMIT || d_acos > ULP_LIMIT {
            failures += 1;
            eprintln!(
                "[{i}] x={:>6.3} | asin ulp={d_asin} (gpu={:e} cpu={exp_asin:e}) | \
                 acos ulp={d_acos} (gpu={:e} cpu={exp_acos:e})",
                xs[i], got_asin[i], got_acos[i],
            );
        }
    }

    for &i in &[0usize, 6, 12] {
        println!(
            "[{i}] x={x:>6.3}  asin gpu={ga} cpu={ea}  acos gpu={gc} cpu={ec}",
            x = xs[i],
            ga = got_asin[i],
            ea = xs[i].asin(),
            gc = got_acos[i],
            ec = xs[i].acos(),
        );
    }

    if failures == 0 {
        println!("\nSUCCESS: {n} cases x 2 functions within {ULP_LIMIT} ULP of host libm");
        Ok(())
    } else {
        eprintln!("\nFAILED: {failures}/{n} cases out of tolerance");
        std::process::exit(1);
    }
}
