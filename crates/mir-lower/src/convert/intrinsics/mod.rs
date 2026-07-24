/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU intrinsic conversion: `dialect-nvvm` → LLVM dialect.
//!
//! This module converts `dialect-nvvm` (NVIDIA Virtual Machine) operations
//! to LLVM dialect ops, using either LLVM NVVM intrinsics or inline PTX
//! assembly depending on the operation.
//!
//! # Lowering Strategies
//!
//! ## Strategy 1: LLVM NVVM Intrinsics
//!
//! For well-supported operations, we call LLVM NVVM intrinsics:
//!
//! ```text
//! nvvm.read_tid_x  →  call i32 @llvm_nvvm_read_ptx_sreg_tid_x()
//! ```
//!
//! This is preferred when available because:
//! - LLVM can optimize the intrinsic
//! - Better debugging information
//! - Portable across LLVM versions
//!
//! ## Strategy 2: Inline PTX Assembly
//!
//! For complex or new operations without LLVM intrinsics, we use inline PTX:
//!
//! ```text
//! wgmma.fence  →  call void asm sideeffect convergent "wgmma.fence.sync.aligned;"
//! ```
//!
//! This is necessary for:
//! - Operations introduced after the LLVM NVVM intrinsics were designed
//! - Operations requiring specific PTX encoding
//! - Performance-critical paths where exact PTX is needed
//!
//! ## The `convergent` Attribute
//!
//! Warp-synchronous operations **must** use the `convergent` attribute on inline
//! assembly. This prevents LLVM from:
//! - Moving the operation across divergent control flow
//! - Speculating the operation
//! - Duplicating the operation
//!
//! Without `convergent`, operations like warp shuffle could be hoisted out of
//! conditionals, breaking the warp-synchronous semantics and causing hangs.
//!
//! # Module Organization
//!
//! | Module       | Description               | Lowering        | Min SM  |
//! |--------------|---------------------------|-----------------|---------|
//! | [`basic`]    | Thread/block IDs, barrier | LLVM intrinsics | All     |
//! | [`debug`]    | Clock, trap, breakpoint   | LLVM intrinsics | All     |
//! | [`warp`]     | Shuffle, vote             | LLVM intrinsics | SM 30+  |
//! | [`cluster`]  | Cluster IDs, DSMEM, sync  | Inline PTX      | SM 90+  |
//! | [`mbarrier`] | Async barriers            | LLVM + PTX      | SM 80+  |
//! | [`wgmma`]    | Warpgroup MMA             | Inline PTX      | SM 90+  |
//! | `tcgen05`    | 5th-gen Tensor Core       | Inline PTX      | SM 100+ |
//! | [`tma`]      | Tensor Memory Access      | Inline PTX      | SM 90+  |
//! | `stmatrix`   | Matrix store              | Inline PTX      | SM 90+  |
//! | [`common`]   | Shared helpers            | -               | -       |
//!
//! # Adding New Intrinsics
//!
//! 1. Add the operation to `dialect-nvvm`
//! 2. Implement `MirToLlvmConversion` for the op in `convert/interface_impls.rs`
//! 3. Implement the conversion function in the appropriate module here
//! 4. Use `call_intrinsic` for LLVM intrinsics or `inline_asm_convergent` for PTX

pub mod asm;
pub mod atomic;
pub mod basic;
pub mod clc;
pub mod cluster;
pub mod common;
pub mod cp_async;
pub mod debug;
pub mod dotprod;
pub mod extended_minmax;
pub mod ldmatrix;
pub mod mbarrier;
pub mod packed;
pub mod prmt;
pub mod scalar_arithmetic;
pub mod scalar_conversion;
pub mod scalar_math;
pub mod tma;
pub mod warp;
pub mod wgmma;
pub mod wmma;
