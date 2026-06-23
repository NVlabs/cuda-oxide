/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Optimization passes over the `dialect-mir` IR.
//!
//! These run in cuda-oxide's middle-end, after `mem2reg` and before lowering to
//! the LLVM dialect. The first is annotation-driven loop unrolling (driven by
//! the `#[unroll]` / `#[unroll(N)]` attribute, carried as a `mir.unroll`
//! attribute on the function op). Future passes (LICM, induction-variable
//! simplification) will live here too.

pub mod unroll;
