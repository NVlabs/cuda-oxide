/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#![feature(f16)]
#![no_std]

pub use cuda_macros::{
    cluster_launch, constant, convergent, cooperative_launch, cuda_module, device, gpu_printf,
    kernel, launch_bounds, pure, readonly,
};

// Re-export for convenience
pub mod atomic;
pub mod barrier;
pub mod clc;
pub mod cluster;
pub mod constant;
pub mod cooperative_groups;
pub mod cusimd;
pub mod debug;
pub mod disjoint;
pub mod fence;
pub mod grid;
pub mod shared;
pub mod tcgen05;
pub mod thread;
pub mod tma;
pub mod warp;
pub mod wgmma;

pub use barrier::{
    // Core type
    Barrier,
    BarrierToken,
    GeneralBarrier,
    Invalidated,
    // Typestate managed barrier
    ManagedBarrier,
    MmaBarrier,
    MmaBarrierHandle,
    Ready,
    // Kind markers
    TmaBarrier,
    TmaBarrier0,
    TmaBarrier1,
    // Type aliases
    TmaBarrierHandle,
    // State markers
    Uninit,
};
pub use constant::{ConstantMemory, ConstantMemoryValue};
pub use cusimd::{CuSimd, Float2, Float4, TmemRegs4, TmemRegs32};
pub use disjoint::DisjointSlice;
pub use fence::*;
pub use shared::{DynamicSharedArray, SharedArray};
pub use tcgen05::{
    TensorMemoryHandle, TmemAddress, TmemDeallocated, TmemF32x4, TmemF32x32, TmemGuard, TmemReady,
    TmemUninit,
};
pub use thread::*;
pub use tma::TmaDescriptor;
