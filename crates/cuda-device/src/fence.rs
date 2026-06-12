/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory fence intrinsics for CUDA device code.
//!
//! These are the device-side visibility primitives used to order ordinary global
//! stores before signaling via atomics:
//!
//! - [`threadfence_block()`] -> PTX `membar.cta`
//! - [`threadfence()`] -> PTX `membar.gl`
//! - [`threadfence_system()`] -> PTX `membar.sys`
//!
//! The functions are compiler-recognized stubs. Their bodies never execute; the
//! cuda-oxide compiler replaces each call with the corresponding NVVM/PTX fence.

/// Block-scoped memory fence.
///
/// Makes the calling thread's prior memory operations visible to threads in the
/// same thread block before any later memory operations can be observed.
///
/// This is equivalent to CUDA C++ `__threadfence_block()`.
#[inline(never)]
pub fn threadfence_block() {
    // Lowered to inline PTX: membar.cta;
    unreachable!("threadfence_block called outside CUDA kernel context")
}

/// Device-scoped memory fence.
///
/// Makes the calling thread's prior global-memory operations visible to other
/// threads on the same GPU before any later memory operations can be observed.
///
/// This is equivalent to CUDA C++ `__threadfence()`.
#[inline(never)]
pub fn threadfence() {
    // Lowered to inline PTX: membar.gl;
    unreachable!("threadfence called outside CUDA kernel context")
}

/// System-scoped memory fence.
///
/// Makes the calling thread's prior global-memory operations visible across the
/// entire system before any later memory operations can be observed. This is
/// the fence needed before publishing a cross-GPU ready flag via a
/// `cuda_device::atomic::SystemAtomicU32`.
///
/// This is equivalent to CUDA C++ `__threadfence_system()`.
#[inline(never)]
pub fn threadfence_system() {
    // Lowered to inline PTX: membar.sys;
    unreachable!("threadfence_system called outside CUDA kernel context")
}
