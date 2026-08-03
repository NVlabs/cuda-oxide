// Copyright (c) 2024-2026 NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scalar `bf16` min/max intrinsics.
//!
//! Each value is carried as the `u16` bit pattern of one bfloat16, matching how
//! [`crate::convert`] already carries packed 16-bit conversion operands. For the
//! SIMD forms that operate on two bfloat16 values in one `u32`, see
//! [`crate::bf16x2`].
//!
//! Only the forms without `ftz` are available. LLVM declares
//! `llvm.nvvm.fmin.ftz.bf16` and its relatives, but the NVPTX backend has no
//! selection pattern for them and instruction selection fails, so they are not
//! exposed here. The `NaN` modifier returns a canonical NaN when either input is
//! NaN instead of the numeric operand, and `xorsign.abs` compares magnitudes and
//! takes the sign from the XOR of the input signs. The plain and `NaN` forms
//! require `sm_80` and PTX ISA 7.0; the `xorsign.abs` forms require `sm_86` and
//! PTX ISA 7.2.

include!("generated/bf16.rs");
