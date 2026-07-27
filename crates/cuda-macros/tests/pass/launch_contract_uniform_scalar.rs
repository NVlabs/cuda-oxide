// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A `Uniform<T>` kernel parameter is marshalled as a bare `T` on the host,
//! because the host is what makes the value uniform: one marshalled value
//! reaches every thread of the launch. The device side receives the witness,
//! so `tile_2d32_rt` needs no `unsafe`.

use cuda_device::{
    DisjointSlice, RuntimeRowMajorTiles, Uniform, cuda_module, kernel, launch_contract, thread,
};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel(launch_context = lc)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn write_cells(
        stride: Uniform<u32>,
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        if let Some(mut cell) = out.tile_2d32_rt(coord, stride) {
            cell.at_const::<0, 0>().write(1.0);
        }
    }

    /// Uniformity is closed under arithmetic whose operands are all uniform,
    /// so a derived stride keeps the witness.
    #[kernel(launch_context = lc)]
    #[launch_contract(domain = 2, coordinates = u32, block = (8, 8, 1))]
    pub fn write_doubled_stride(
        stride: Uniform<u32>,
        mut out: DisjointSlice<f32, RuntimeRowMajorTiles<1, 1>>,
    ) {
        let coord = thread::coord_2d_u32(lc);
        let doubled = stride.wrapping_mul_const::<2>();
        if let Some(mut cell) = out.tile_2d32_rt(coord, doubled) {
            cell.at_const::<0, 0>().write(2.0);
        }
    }
}

/// The generated host method takes a plain `u32`, not a `Uniform<u32>`.
fn host_signature_takes_the_bare_scalar(
    module: &kernels::LoadedModule,
    stream: &cuda_core::CudaStream,
    out: &mut cuda_core::DeviceBuffer<f32>,
) -> Result<(), cuda_core::LaunchContractError> {
    let prepared = module.prepare_write_cells(cuda_core::LaunchConfig2D::new((1, 1), (8, 8), 0))?;
    module.write_cells(stream, &prepared, 64u32, out)
}

fn main() {
    let _ = host_signature_takes_the_bare_scalar;
}
