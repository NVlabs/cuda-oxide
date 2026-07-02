/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Checked arithmetic smoke test.
//!
//! Verifies that `overflowing_add`, `overflowing_sub`, and `overflowing_mul`
//! return the correct (result, overflow) pair. Before the fix, the overflow
//! flag was hardcoded to `false`; now it is computed by the proper LLVM
//! overflow intrinsics.
//!
//! Each kernel encodes both the wrapping result (low byte) and the overflow
//! flag (bit 8) into a single u32 output so the host can check both in one
//! pass.
//!
//! Run: cargo oxide run checked_arith

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    /// out[i] = (a[i].overflowing_add(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_add(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_add(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }

    /// out[i] = (a[i].overflowing_sub(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_sub(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_sub(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }

    /// out[i] = (a[i].overflowing_mul(b[i]).0 as u32) | ((overflow as u32) << 8)
    #[kernel]
    pub fn checked_mul(a: &[u8], b: &[u8], mut out: DisjointSlice<u32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(o) = out.get_mut(idx) {
            let (result, overflow) = a[i].overflowing_mul(b[i]);
            *o = (result as u32) | ((overflow as u32) << 8);
        }
    }
}

fn check(label: &str, got: u32, expected_result: u8, expected_overflow: bool) -> bool {
    let got_result = (got & 0xff) as u8;
    let got_overflow = (got >> 8) & 1 == 1;
    if got_result != expected_result || got_overflow != expected_overflow {
        eprintln!(
            "  FAIL {label}: result={got_result} (want {expected_result}), \
             overflow={got_overflow} (want {expected_overflow})"
        );
        false
    } else {
        true
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("load module");
    let cfg = LaunchConfig::for_num_elems(4);

    // --- overflowing_add ---
    // 200 + 100 = 300 -> wraps to 44, overflow
    // 100 + 50  = 150 -> no overflow
    // 255 + 1   = 256 -> wraps to 0, overflow
    // 0   + 0   = 0   -> no overflow
    let a_add: Vec<u8> = vec![200, 100, 255, 0];
    let b_add: Vec<u8> = vec![100, 50, 1, 0];
    let a_dev = DeviceBuffer::from_host(&stream, &a_add).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_add).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_add(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_add launch");
    let out_add = out_dev.to_host_vec(&stream).unwrap();

    // --- overflowing_sub ---
    // 100 - 200 = -100 -> wraps to 156, overflow
    // 200 - 100 = 100  -> no overflow
    // 0   - 1   = -1   -> wraps to 255, overflow
    // 50  - 50  = 0    -> no overflow
    let a_sub: Vec<u8> = vec![100, 200, 0, 50];
    let b_sub: Vec<u8> = vec![200, 100, 1, 50];
    let a_dev = DeviceBuffer::from_host(&stream, &a_sub).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_sub).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_sub(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_sub launch");
    let out_sub = out_dev.to_host_vec(&stream).unwrap();

    // --- overflowing_mul ---
    // 20  * 10  = 200   -> no overflow
    // 20  * 20  = 400   -> wraps to 144, overflow
    // 255 * 2   = 510   -> wraps to 254, overflow
    // 1   * 1   = 1     -> no overflow
    let a_mul: Vec<u8> = vec![20, 20, 255, 1];
    let b_mul: Vec<u8> = vec![10, 20, 2, 1];
    let a_dev = DeviceBuffer::from_host(&stream, &a_mul).unwrap();
    let b_dev = DeviceBuffer::from_host(&stream, &b_mul).unwrap();
    let mut out_dev = DeviceBuffer::<u32>::zeroed(&stream, 4).unwrap();
    // SAFETY: launch shape/resources match the kernel; buffers cover its accesses.
    unsafe { module.checked_mul(&stream, cfg, &a_dev, &b_dev, &mut out_dev) }
        .expect("checked_mul launch");
    let out_mul = out_dev.to_host_vec(&stream).unwrap();

    let mut ok = true;
    ok &= check("add[0] 200+100", out_add[0], 44, true);
    ok &= check("add[1] 100+50", out_add[1], 150, false);
    ok &= check("add[2] 255+1", out_add[2], 0, true);
    ok &= check("add[3] 0+0", out_add[3], 0, false);

    ok &= check("sub[0] 100-200", out_sub[0], 156, true);
    ok &= check("sub[1] 200-100", out_sub[1], 100, false);
    ok &= check("sub[2] 0-1", out_sub[2], 255, true);
    ok &= check("sub[3] 50-50", out_sub[3], 0, false);

    ok &= check("mul[0] 20*10", out_mul[0], 200, false);
    ok &= check("mul[1] 20*20", out_mul[1], 144, true);
    ok &= check("mul[2] 255*2", out_mul[2], 254, true);
    ok &= check("mul[3] 1*1", out_mul[3], 1, false);

    if ok {
        println!("SUCCESS: all overflowing_{{add,sub,mul}} results correct");
    } else {
        std::process::exit(1);
    }
}
