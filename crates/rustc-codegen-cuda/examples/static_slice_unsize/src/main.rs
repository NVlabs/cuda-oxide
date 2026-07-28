/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Zero-addend device-static array→slice unsize.
//!
//! `const TABLE_SLICE: &[f32] = &TABLE` coerces `&[f32; 4]` to a fat `&[f32]`.
//! The importer materializes the thin global pointer plus the array length.
//!
//! Run: `cargo oxide run static_slice_unsize`

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::kernel;
use cuda_host::cuda_module;

static TABLE: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// `&TABLE` is `&[f32; 4]`; the unsize coercion to `&[f32]` keeps a zero
/// addend and adds the length metadata a thin static pointer cannot carry.
const TABLE_SLICE: &[f32] = &TABLE;

#[inline(never)]
fn table_slice() -> &'static [f32] {
    TABLE_SLICE
}

#[cuda_module]
mod kernels {
    use super::*;

    /// # Safety
    ///
    /// `out` must point to device-accessible storage that is properly aligned
    /// and writable for one `f32`. No other thread may race with this write.
    #[kernel]
    pub unsafe fn slice_unsize(out: *mut f32) {
        let table = table_slice();
        unsafe {
            *out = table[0] + table[3];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).expect("create CUDA context");
    let stream = ctx.default_stream();
    // SAFETY: this package owns the embedded device bundle for `kernels`.
    let module = unsafe { kernels::load(&ctx).expect("load module") };

    let mut out = DeviceBuffer::<f32>::zeroed(&stream, 1).expect("alloc out");
    // SAFETY: one-thread launch writing a single f32.
    unsafe {
        module
            .slice_unsize(&stream, LaunchConfig::for_num_elems(1), out.as_mut_ptr())
            .expect("launch");
    }
    stream.synchronize().expect("sync");

    let host = out.to_host_vec(&stream).expect("dtoh");
    let expected = TABLE[0] + TABLE[3];
    assert!(
        (host[0] - expected).abs() < 1e-6,
        "got {} expected {}",
        host[0],
        expected
    );
    println!("static_slice_unsize: PASS ({})", host[0]);
}
