// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A caller-defined slice view borrows the buffer it reports, so it cannot be
//! launched after that buffer is gone. The generated launcher takes the view
//! by reference and the borrow checker rejects the escaped borrow.

use cuda_device::{cuda_module, kernel};
use cuda_host::{KernelSliceArg, KernelSliceArgMut};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn fill(out: &mut [f32]) {
        let _ = out;
    }
}

struct View<'a> {
    buffer: &'a mut cuda_core::DeviceBuffer<f32>,
}

// SAFETY: the view forwards the pointer and length of the buffer it borrows.
unsafe impl KernelSliceArg for View<'_> {
    type Elem = f32;
    fn cu_deviceptr(&self) -> cuda_core::sys::CUdeviceptr {
        self.buffer.cu_deviceptr()
    }
    fn len(&self) -> usize {
        self.buffer.len()
    }
}
unsafe impl KernelSliceArgMut for View<'_> {}

fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    ctx: &std::sync::Arc<cuda_core::CudaContext>,
) {
    let cfg = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut view = {
        // SAFETY: never executed; this fixture only has to fail to compile.
        let mut local =
            unsafe { cuda_core::DeviceBuffer::<f32>::from_raw_parts(0, 0, ctx.clone()) };
        View { buffer: &mut local }
    };
    let _ = unsafe { module.fill(stream, cfg, &mut view) };
}

fn main() {}
