/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Type conversion intrinsics.
//!
//! These intrinsics provide access to PTX type conversion instructions that
//! are more efficient than scalar Rust casts.

include!("generated/convert.rs");
