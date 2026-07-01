/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared state types for `dialect-mir` → LLVM dialect lowering.
//!
//! The DialectConversion framework handles value mapping and block mapping
//! automatically. This module provides the CUDA-specific state types that
//! certain ops need during conversion.

use rustc_hash::FxHashMap;

/// Map from shared memory allocation keys to their LLVM global symbol names.
///
/// In CUDA kernels, shared memory is declared as module-level globals with
/// address space 3. When multiple operations reference the same shared allocation
/// (identified by a key string), they should all refer to the same global.
pub type SharedGlobalsMap = FxHashMap<String, pliron::identifier::Identifier>;

/// Map from ordinary device static keys to LLVM global symbol names.
///
/// Ordinary Rust `static` / `static mut` values used from device code live in
/// CUDA global memory (address space 1), not shared memory.
pub type DeviceGlobalsMap = FxHashMap<String, pliron::identifier::Identifier>;

/// Tracking for dynamic shared memory alignment per lowered function.
///
/// Maps function name to `(symbol_name, max_alignment)`.
///
/// Each function that owns a dynamic shared-memory access gets a symbol. Before
/// conversion, the pass combines the alignment requested by the function body
/// with every propagated launch-contract marker that can reach it. This
/// ensures a helper shared by several kernels uses their strongest requirement.
pub type DynamicSmemAlignmentMap = FxHashMap<String, (pliron::identifier::Identifier, u64)>;
