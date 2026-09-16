/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory owned by cudarc as a kernel argument: a newtype over `CudaSlice`
//! implements the slice contracts, so the generated launcher takes it as is.
//! No copy into a `DeviceBuffer`, no ownership transfer.
//!
//!   cargo oxide run cudarc_slice

use cuda_core::CudaContext;
use cuda_core::simt::LaunchConfig;
use cuda_device::{DisjointSlice, cuda_module, kernel, thread};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};
use cudarc::driver::{CudaSlice, DevicePtr};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn double(input: &[f32], mut output: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out) = output.get_mut(idx) {
            *out = 2.0 * input[i];
        }
    }
}

/// A cudarc allocation, launched by the oxide typed API.
struct Cudarc<T>(CudaSlice<T>);

// SAFETY: the slice owns a live allocation of `len` elements for as long as
// the wrapper does; `&mut` is the write authority.
unsafe impl<T> KernelSliceArg for Cudarc<T> {
    type Elem = T;
    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.0.device_ptr(self.0.stream()).0
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}
unsafe impl<T> KernelSliceArgMut for Cudarc<T> {}

fn main() {
    // cudarc allocates and uploads on its default (null) stream.
    let cudarc = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = cudarc.default_stream();
    let input = Cudarc(stream.clone_htod(&[1.0f32, 2.0, 3.0, 4.0]).unwrap());
    let mut output = Cudarc(stream.alloc_zeros::<f32>(4).unwrap());

    // oxide launches on the same primary context and the same null stream.
    let ctx = CudaContext::new(0).unwrap();
    let module = kernels::load(&ctx).unwrap();
    // SAFETY: four threads over four elements; one stream orders everything.
    unsafe {
        module.double(
            &ctx.default_stream(),
            LaunchConfig::for_num_elems(4),
            &input,
            &mut output,
        )
    }
    .unwrap();

    let result = stream.clone_dtoh(&output.0).unwrap();
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
    println!("SUCCESS: cudarc memory doubled in place by an oxide kernel: {result:?}");
}
