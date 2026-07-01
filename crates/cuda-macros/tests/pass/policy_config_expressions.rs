// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

use cuda_device::{cuda_module, device, kernel, launch_bounds, launch_contract};

trait Policy {
    const MAX_THREADS: u32;
    const MIN_BLOCKS: u32;
    const UNROLL: u32;
}

#[kernel]
#[launch_bounds(P::MAX_THREADS * 2, P::MIN_BLOCKS)]
fn configured<P: Policy>() {
    let mut index = 0;
    #[unroll(P::UNROLL)]
    while index < 8 {
        index += 1;
    }
}

#[device]
fn configured_helper<P: Policy>() {
    let mut index = 0;
    #[unroll(P::UNROLL)]
    while index < 8 {
        index += 1;
    }
}

#[kernel]
fn function_local_policy_value() {
    const FACTOR: u32 = 4;
    let mut index = 0;
    #[unroll(FACTOR)]
    while index < 8 {
        index += 1;
    }
}

#[cuda_module]
mod contracted {
    use super::*;

    #[kernel]
    #[launch_bounds(256, P::MIN_BLOCKS)]
    #[launch_contract(domain = 1)]
    pub fn configured<P: Policy>() {}
}

fn main() {}
