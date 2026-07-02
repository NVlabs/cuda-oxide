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

use crate::LoweringOptions;
use pliron::context::Context;

mod options_storage {
    pliron::dict_key!(LOWERING_OPTIONS_KEY, "cuda_oxide_mir_lower_options");
}

/// Store the options for the active lowering pass in pliron's per-compilation
/// context. Conversion interfaces only receive the context, so this keeps
/// policy explicit without consulting process-global environment variables in
/// individual operation converters.
pub(crate) fn set_lowering_options(ctx: &mut Context, options: LoweringOptions) {
    if let Some(index) = ctx
        .aux_data_map
        .get(&*options_storage::LOWERING_OPTIONS_KEY)
        .copied()
    {
        ctx.aux_data[index] = Box::new(options);
    } else {
        let index = ctx.aux_data.insert(Box::new(options));
        ctx.aux_data_map
            .insert(options_storage::LOWERING_OPTIONS_KEY.clone(), index);
    }
}

/// Read options for the active lowering pass.
///
/// The default preserves the historical behavior for callers that use the
/// original `lower_mir_to_llvm` entry point.
pub(crate) fn lowering_options(ctx: &Context) -> LoweringOptions {
    ctx.aux_data_map
        .get(&*options_storage::LOWERING_OPTIONS_KEY)
        .and_then(|index| ctx.aux_data[*index].downcast_ref::<LoweringOptions>())
        .copied()
        .unwrap_or_default()
}

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
pub type DynamicSmemAlignmentMap = FxHashMap<String, (pliron::identifier::Identifier, u64)>;
