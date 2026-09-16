// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A `RowWidth` binding is not a slice view. A kernel whose parameter is a
//! plain `&mut [T]` takes `&mut impl KernelSliceArgMut`, and `RowWidth` does
//! not implement that contract on purpose: accepting it would silently drop
//! the bound width. The call must not compile.

use cuda_device::{cuda_module, kernel};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn flat(out: &mut [f32]) {
        let _ = out;
    }
}

fn launch(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    out: &mut cuda_core::DeviceBuffer<f32>,
) {
    let cfg = cuda_core::simt::LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bound = cuda_host::RowWidth::new(out, 64);
    let _ = unsafe { module.flat(stream, cfg, &mut bound) };
}

fn main() {}
