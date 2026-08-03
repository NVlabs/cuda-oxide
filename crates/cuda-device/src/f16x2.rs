// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Packed `f16x2` arithmetic intrinsics.
//!
//! Each `u32` stores two f16 values. The first value uses the low 16 bits.
//! The second value uses the high 16 bits.
//!
//! The packing here is the ALU format for SIMD arithmetic. For over-aligned
//! *memory element* types that make loads and stores single wide transactions
//! (e.g. eight packed halves moving as one 128-bit access), see
//! [`crate::vector`]. For multi-register *value* groups, see
//! [`crate::cusimd::CuSimd`].
//!
//! See also [`crate::f16`] for the scalar min/max forms that operate on a
//! single half rather than a packed pair.

include!("generated/f16x2.rs");
