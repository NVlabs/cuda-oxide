// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use cuda_device::thread::{Index2D, Runtime2DIndex};
use cuda_device::{DisjointSlice, cuda_module, kernel, launch_bounds, launch_contract};

type Tiled = Index2D<64>;
use cuda_device::thread::Index2D as ImportedIndex2D;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    #[launch_bounds(64)]
    #[launch_contract(domain = 2, block = (8, 8, 1))]
    pub fn type_alias(mut out: DisjointSlice<u32, Tiled>) {
        let _ = &mut out;
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 8, 1))]
    pub fn import_alias(mut out: DisjointSlice<u32, ImportedIndex2D<64>>) {
        let _ = &mut out;
    }

    #[kernel]
    #[launch_contract(domain = 2, block = (8, 8, 1))]
    pub fn runtime_2d(mut out: DisjointSlice<u32, Runtime2DIndex>) {
        let _ = &mut out;
    }

    // A 2D index formula is also sound for a 1D launch because every Y
    // dimension is constrained to one.
    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn two_dimensional_index_on_1d_launch(mut out: DisjointSlice<u32, Tiled>) {
        let _ = &mut out;
    }

    #[kernel]
    #[launch_contract(domain = 1, block = (64, 1, 1))]
    pub fn lifetime_only<'a>(input: &'a [u32]) {
        let _ = input;
    }
}

fn prepared_methods_exist(module: &kernels::LoadedModule) {
    let _ = module.prepare_type_alias(cuda_core::LaunchConfig2D::new((1, 1), (8, 8), 0));
    let _ = module.prepare_import_alias(cuda_core::LaunchConfig2D::new((1, 1), (8, 8), 0));
    let _ = module.prepare_runtime_2d(cuda_core::LaunchConfig2D::new((1, 1), (8, 8), 0));
    let _ = module.prepare_two_dimensional_index_on_1d_launch(
        cuda_core::LaunchConfig1D::new(1, 64, 0),
    );
    let _ = module.prepare_lifetime_only(cuda_core::LaunchConfig1D::new(1, 64, 0));
}

fn main() {}
