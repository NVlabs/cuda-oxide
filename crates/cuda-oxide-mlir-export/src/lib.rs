/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! CUDA Oxide mappings for textual MLIR export.
//!
//! The generic exporter answers "how do we build typed, deterministic MLIR
//! text?" This crate answers "what does this CUDA Oxide operation become for
//! this exact consumer?"

mod cute;
mod cute_gemm;
mod cute_gemv;
mod mir_core;
mod mir_memory;
mod nvvm_sregs;
mod packs;
mod profile;

pub use cute::register_cute_elementwise_pack;
pub(crate) use cute_gemm::register_cute_gemm_pack;
pub(crate) use cute_gemv::register_cute_cutlass_profile_pack;
pub use mir_core::register_mir_core_pack;
pub use nvvm_sregs::register_nvvm_sreg_pack;
pub use packs::register_builtin_pack;
pub use profile::{CutlassFullCuteMlir22, ExportManifest, MlirConsumerProfile, ProfileError};
