/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU intrinsic dispatch and expansion.
//!
//! This module handles the translation of `cuda_device` intrinsic calls into
//! `dialect-nvvm` operations. Intrinsics are organized by functional category:
//!
//! | Module      | Intrinsics                                                                   |
//! |-------------|------------------------------------------------------------------------------|
//! | `indexing`  | `threadIdx_*`, `blockIdx_*`, `index_1d`, `index_2d::<S>`, `index_2d_runtime` |
//! | `sync`      | `sync_threads`, `mbarrier_*`, `fence_*`                                      |
//! | `cluster`   | `cluster_ctaidX`, `cluster_sync`, `map_shared_rank`                          |
//! | `warp`      | `shuffle_*`, `vote_*`, `lane_id`                                             |
//! | `wgmma`     | Hopper WGMMA matrix operations                                               |
//! | `tcgen05`   | Blackwell tensor core (tcgen05) operations                                   |
//! | `tma`       | Tensor Memory Access (TMA) operations                                        |
//! | `memory`    | `SharedArray`, `stmatrix_*`, type conversions                                |
//! | `debug`     | `clock`, `clock64`, `globaltimer`, `trap`, `breakpoint`                      |
//!
//! # Architecture
//!
//! Each intrinsic module exports `emit_*` functions that:
//! 1. Take MIR operands and translate them to pliron IR values
//! 2. Create the appropriate `dialect-nvvm` operations
//! 3. Store results in the value map
//! 4. Emit a zero-operand `mir.goto` to the call's single successor block
//!
//! # Note
//!
//! Currently, all emit functions remain in `terminator/mod.rs` for compilation
//! stability. This module structure is prepared for gradual migration of
//! functions to their respective category modules.

// Submodules for intrinsic categories (to be populated incrementally)
pub mod atomic;
pub mod bigint;
pub mod bitops;
pub mod clc;
pub mod cluster;
pub mod debug;
pub mod float_math;
pub mod indexing;
pub mod memory;
pub mod saturating;
pub mod sync;
pub mod tcgen05;
pub mod tma;
pub mod warp;
pub mod wgmma;
