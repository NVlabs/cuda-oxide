/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! `cvt_rn_satfinite_e4m3x2_f32` intrinsic: pack an f32 pair into a u16
//! holding two e4m3 FP8 values (sm_89+).
//!
//! The kernel packs (lo, hi) f32 pairs into u16 e4m3x2 values via a single
//! `cvt.rn.satfinite.e4m3x2.f32` PTX instruction. The host verifies
//! bit-exact agreement with a scalar round-to-nearest-even reference
//! (nearest e4m3 code, ties to even mantissa, satfinite clamp to ±448),
//! a constant-literal kernel that pins the lane order, and a NaN launch
//! that checks both lanes carry the e4m3 NaN pattern.
//!
//! Run: cargo oxide run cvt_e4m3x2

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::convert::cvt_rn_satfinite_e4m3x2_f32;
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn pack_e4m3x2(lo_in: &[f32], hi_in: &[f32], mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        let i = idx.get() as usize;
        if let Some(out_elem) = out.get_mut(idx) {
            let lo = lo_in[i];
            let hi = hi_in[i];
            *out_elem = cvt_rn_satfinite_e4m3x2_f32(lo, hi);
        }
    }

    /// Edge case: constant literal operands (no locals involved).
    #[kernel]
    pub fn pack_e4m3x2_consts(mut out: DisjointSlice<u16>) {
        let idx = thread::index_1d();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = cvt_rn_satfinite_e4m3x2_f32(1.5_f32, -2.25_f32);
        }
    }
}

/// Decode a positive-magnitude e4m3 code (0x00..=0x7E) to f64.
///
/// e4m3 (OCP E4M3FN): bias 7, exponent 0 is subnormal, code 0x7F is NaN,
/// no infinities; max finite is 0x7E = 448.
fn e4m3_to_f64(code: u8) -> f64 {
    let e = (code >> 3) & 0xF;
    let m = (code & 7) as f64;
    if e == 0 {
        m / 8.0 * 2f64.powi(-6)
    } else {
        (1.0 + m / 8.0) * 2f64.powi(e as i32 - 7)
    }
}

/// Scalar reference for `cvt.rn.satfinite.e4m3x2.f32` on one lane.
///
/// Nearest finite e4m3 magnitude with ties to even mantissa. Because the
/// search only covers finite codes, overflow clamps to ±448 (0x7E),
/// which is exactly what `satfinite` does. All arithmetic is exact in
/// f64 (f32 inputs and e4m3 values are both exactly representable).
fn f32_to_e4m3_satfinite(x: f32) -> u8 {
    assert!(!x.is_nan(), "NaN lanes are checked separately");
    let sign = if x.is_sign_negative() { 0x80 } else { 0x00 };
    let a = f64::from(x.abs());

    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for code in 0..=0x7Eu8 {
        let err = (e4m3_to_f64(code) - a).abs();
        if err < best_err || (err == best_err && code % 2 == 0 && best % 2 == 1) {
            best = code;
            best_err = err;
        }
    }
    sign | best
}

fn scalar_ref(lo: f32, hi: f32) -> u16 {
    u16::from(f32_to_e4m3_satfinite(lo)) | (u16::from(f32_to_e4m3_satfinite(hi)) << 8)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 256;
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    let (major, minor) = ctx.compute_capability()?;
    println!("GPU Compute Capability: sm_{}{}", major, minor);

    // FP8 conversions require sm_89+ (Ada Lovelace).
    if (major, minor) < (8, 9) {
        println!("skipping: cvt.rn.satfinite.e4m3x2.f32 requires sm_89+");
        return Ok(());
    }

    let module = ctx.load_module_from_file("cvt_e4m3x2.ptx")?;
    let module = kernels::from_module(module)?;

    // lo sweeps past ±448 to exercise the satfinite clamp; hi sweeps the
    // subnormal range around zero (min subnormal is 2^-9 ≈ 0.00195).
    let lo_host: Vec<f32> = (0..N).map(|i| (i as f32 - 128.0) * 4.7).collect();
    let hi_host: Vec<f32> = (0..N).map(|i| (i as f32) * 0.0031 - 0.4).collect();

    let lo_dev = DeviceBuffer::from_host(&stream, &lo_host)?;
    let hi_dev = DeviceBuffer::from_host(&stream, &hi_host)?;
    let mut out_dev = DeviceBuffer::<u16>::zeroed(&stream, N)?;

    let cfg = LaunchConfig::for_num_elems(N as u32);
    module.pack_e4m3x2(&stream, cfg, &lo_dev, &hi_dev, &mut out_dev)?;

    let out_host = out_dev.to_host_vec(&stream)?;
    let mut failures = 0;
    for i in 0..N {
        let expect = scalar_ref(lo_host[i], hi_host[i]);
        if out_host[i] != expect {
            if failures < 5 {
                eprintln!(
                    "MISMATCH i={i}: lo={} hi={} got={:#06x} want={:#06x}",
                    lo_host[i], hi_host[i], out_host[i], expect
                );
            }
            failures += 1;
        }
    }

    // Constant-literal kernel: lane order with known bit patterns.
    // e4m3(1.5) = 0x3C (low byte), e4m3(-2.25) = 0xC1 (high byte).
    let mut out_const_dev = DeviceBuffer::<u16>::zeroed(&stream, 1)?;
    let cfg1 = LaunchConfig::for_num_elems(1);
    module.pack_e4m3x2_consts(&stream, cfg1, &mut out_const_dev)?;
    let got_const = out_const_dev.to_host_vec(&stream)?[0];
    if got_const != 0xC13C {
        eprintln!("CONST MISMATCH: got={got_const:#06x} want=0xc13c");
        failures += 1;
    }

    // NaN inputs must produce the e4m3 NaN pattern (all-ones magnitude,
    // 0x7f) in both lanes; satfinite must not clamp NaN to ±448.
    let nan_host = vec![f32::NAN; 1];
    let nan_dev = DeviceBuffer::from_host(&stream, &nan_host)?;
    let mut out_nan_dev = DeviceBuffer::<u16>::zeroed(&stream, 1)?;
    module.pack_e4m3x2(&stream, cfg1, &nan_dev, &nan_dev, &mut out_nan_dev)?;
    let got_nan = out_nan_dev.to_host_vec(&stream)?[0];
    let [nan_lo, nan_hi] = got_nan.to_le_bytes();
    if nan_lo & 0x7F != 0x7F || nan_hi & 0x7F != 0x7F {
        eprintln!("NAN MISMATCH: got={got_nan:#06x}, both lanes must match 0x7f magnitude");
        failures += 1;
    }

    if failures == 0 {
        println!(
            "SUCCESS: all {N} packed e4m3x2 values match scalar reference (+ const lane-order and NaN checks)"
        );
        Ok(())
    } else {
        Err(format!("{failures} mismatches").into())
    }
}
