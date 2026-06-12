/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared state types for `dialect-mir` → LLVM dialect lowering.
//!
//! The DialectConversion framework handles value mapping and block mapping
//! automatically. This module provides the CUDA-specific state types that
//! certain ops need during conversion.

use std::collections::HashMap;

/// Map from shared memory allocation keys to their LLVM global symbol names.
///
/// In CUDA kernels, shared memory is declared as module-level globals with
/// address space 3. When multiple operations reference the same shared allocation
/// (identified by a key string), they should all refer to the same global.
pub type SharedGlobalsMap = HashMap<String, pliron::identifier::Identifier>;

/// Map from ordinary device static keys to LLVM global symbol names.
///
/// Ordinary Rust `static` / `static mut` values used from device code live in
/// CUDA global memory (address space 1), not shared memory.
pub type DeviceGlobalsMap = HashMap<String, pliron::identifier::Identifier>;

/// Tracking for dynamic shared memory alignment per kernel.
///
/// Maps kernel name to `(symbol_name, max_alignment)`.
///
/// Each kernel gets its own symbol (e.g., `__dynamic_smem_my_kernel`)
/// for explicit separation in the generated PTX. Before converting any
/// operations, the pass pre-scans all `MirExternSharedOp` operations in
/// a function to determine the maximum alignment required by any
/// `DynamicSharedArray<T, ALIGN>` call, ensuring the global is created
/// with the correct alignment from the start.
pub type DynamicSmemAlignmentMap = HashMap<String, (pliron::identifier::Identifier, u64)>;
