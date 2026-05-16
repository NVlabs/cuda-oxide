/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! NVVM dialect operations.
//!
//! This module defines Pliron IR operations for the NVVM (NVIDIA Virtual Machine) dialect,
//! which maps directly to LLVM's NVPTX backend intrinsics for GPU execution.
//!
//! # Module Organization
//!
//! Operations are organized by functional category:
//!
//! ```text
//! ┌─────────────┬────────────────────────────────────┬────────────┬──────┐
//! │ Module      │ Description                        │ GPU Arch   │ Ops  │
//! ├─────────────┼────────────────────────────────────┼────────────┼──────┤
//! │ atomic      │ Atomic load/store/RMW/CAS          │ sm_70+     │ 4    │
//! │ thread      │ Thread/block/grid indexing         │ All        │ 7    │
//! │ warp        │ Warp shuffle and vote operations   │ All        │ 12   │
//! │ cluster     │ Thread Block Cluster ops + DSMEM   │ Hopper+    │ 10   │
//! │ mbarrier    │ Async barrier (mbarrier) ops       │ Hopper+    │ 9    │
//! │ tma         │ Tensor Memory Accelerator ops      │ Hopper+    │ 13   │
//! │ wgmma       │ Warpgroup Matrix Multiply-Acc      │ Hopper+    │ 5    │
//! │ tcgen05     │ Tensor Core Gen 5 operations       │ Blackwell+ │ 25+  │
//! │ stmatrix    │ Shared memory matrix store         │ Hopper+    │ 5    │
//! └─────────────┴────────────────────────────────────┴────────────┴──────┘
//! ```
//!
//! # Architecture Requirements
//!
//! - **All GPUs**: `thread`, basic `warp` operations
//! - **Hopper+ (sm_90+)**: `cluster`, `mbarrier`, `tma`, `wgmma`
//! - **Blackwell+ (sm_100+)**: `tcgen05`
//!
//! # Verification Strategy
//!
//! NVVM operations use **minimal structural verification** (operand/result counts only),
//! deliberately omitting detailed type verification. This is a conscious design decision:
//!
//! ## Why Minimal Verification?
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────┐
//! │ Data Flow: User Rust Code → rustc → MIR → mir-importer → mir-lower → NVVM  │
//! │                               ↑                            ↑          ↓    │
//! │                          Type-safe                    Our code    LLVM     │
//! │                                                                  verifies  │
//! └─────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! 1. **NVVM ops are machine-generated**: All NVVM ops are produced by `mir-lower`,
//!    not written by users. Type errors in user code are caught by `rustc` before
//!    we ever see them.
//!
//! 2. **LLVM provides downstream verification**: These ops are thin wrappers around
//!    LLVM intrinsics. When lowered to LLVM IR, the LLVM verifier catches type
//!    mismatches with acceptable error messages.
//!
//! 3. **Operand count is the common error**: The most likely bug in code generation
//!    is passing the wrong number of arguments. This is already verified.
//!
//! 4. **Avoids 1500+ lines of boilerplate**: Full type verification for 70+ ops
//!    would add significant code with marginal benefit.
//!
//! ## What IS Verified
//!
//! - **Operand count**: Each op verifies it has the correct number of operands.
//! - **Result count**: Each op verifies it produces the correct number of results.
//! - **Thread indexing ops**: Verify result is `i32` (these are the simplest and
//!   most commonly used, so the extra check is worthwhile).
//! - **Tcgen05 pure loads**: Verify exact result count (32 or 4 results).
//!
//! ## What is NOT Verified (by design)
//!
//! - Operand types (e.g., that shuffle value is i32/f32)
//! - Pointer address spaces (e.g., that mbarrier ptr is addrspace(3))
//! - Descriptor types (e.g., that TMA descriptor is i64)
//!
//! These are validated at LLVM lowering time when the intrinsic calls are generated.
//!
//! ## When to Add Type Verification
//!
//! Consider adding type verification if:
//! - NVVM ops become user-constructible (e.g., via a DSL)
//! - Error messages from LLVM lowering prove inadequate in practice
//! - A specific op is frequently misused and earlier detection is valuable
//!
//! # Usage
//!
//! All operations are re-exported at this module level for convenience:
//!
//! ```ignore
//! use dialect_nvvm::ops::{ReadPtxSregTidXOp, Barrier0Op, ShflSyncBflyI32Op};
//! ```

pub mod atomic;
mod clc;
mod cluster;
mod convert;
mod debug;
mod grid;
mod mbarrier;
mod stmatrix;
mod tcgen05;
mod thread;
mod tma;
mod warp;
mod wgmma;

use pliron::context::Context;

// Re-export all operations for public API
pub use atomic::*;
pub use clc::*;
pub use cluster::*;
pub use convert::*;
pub use debug::*;
pub use grid::*;
pub use mbarrier::*;
pub use stmatrix::*;
pub use tcgen05::*;
pub use thread::*;
pub use tma::*;
pub use warp::*;
pub use wgmma::*;

/// Register all NVVM dialect operations with the context.
///
/// This function registers all operation types so they can be parsed,
/// verified, and printed. Must be called during dialect initialization.
pub fn register(ctx: &mut Context) {
    atomic::register(ctx);
    clc::register(ctx);
    convert::register(ctx);
    thread::register(ctx);
    warp::register(ctx);
    cluster::register(ctx);
    grid::register(ctx);
    mbarrier::register(ctx);
    tma::register(ctx);
    wgmma::register(ctx);
    tcgen05::register(ctx);
    stmatrix::register(ctx);
    debug::register(ctx);
}
