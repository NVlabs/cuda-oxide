// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two f16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.

include!("generated/f16x2.rs");
