// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scalar `f16` min/max intrinsics.
//!
//! Each value is carried as the `u16` bit pattern of one IEEE half, matching
//! how [`crate::convert`] already carries packed 16-bit conversion operands.
//! For the SIMD forms that operate on two halves in one `u32`, see
//! [`crate::f16x2`].
//!
//! The `ftz`, `NaN`, and `xorsign.abs` modifiers select PTX behaviour directly:
//! `ftz` flushes subnormal inputs and results to zero, `NaN` returns a canonical
//! NaN when either input is NaN instead of the numeric operand, and
//! `xorsign.abs` compares magnitudes and takes the sign from the XOR of the
//! input signs. The plain and `NaN` forms require `sm_80` and PTX ISA 7.0; every
//! `xorsign.abs` form requires `sm_86` and PTX ISA 7.2.

include!("generated/f16.rs");
