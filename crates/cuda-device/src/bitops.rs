/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Bit-manipulation PTX operations.
//!
//! This module exposes per-thread PTX instructions that operate on raw
//! bit patterns.
//!
//! # Operations
//!
//! | Function   | PTX Instruction   | Description                       |
//! |------------|-------------------|-----------------------------------|
//! | `prmt_b32` | `prmt.b32`        | Byte permute on two 32-bit words  |
//!
//! # `prmt.b32` — Byte Permute
//!
//! Selects four bytes from two 32-bit inputs according to a control word.
//!
//! The two inputs `a` (bits `[63:32]`) and `b` (bits `[31:0]`) form an
//! 8-byte source array. The control word `c` contains four 4-bit selectors
//! (one per output byte) that choose which source byte goes to each output
//! byte position.
//!
//! ```text
//! Source bytes:   a[3] a[2] a[1] a[0] b[3] b[2] b[1] b[0]
//!                   7    6    5    4    3    2    1    0
//!
//! Control word c: [sel3 | sel2 | sel1 | sel0]  (4 bits each)
//!
//! Result byte i = source[sel_i & 0x7]
//! ```
//!
//! # Requirements
//!
//! - **PTX ISA**: 2.0+
//! - **Architecture**: sm_20+ (all modern GPUs)

/// Byte permute: rearrange bytes from two 32-bit words.
///
/// Selects four bytes from the concatenation of `a` (high) and `b` (low)
/// according to the control word `c`, producing a new 32-bit value.
///
/// # Arguments
///
/// - `a` — upper 32 bits of the source (bytes 7..4)
/// - `b` — lower 32 bits of the source (bytes 3..0)
/// - `c` — control word with four 4-bit byte selectors
///
/// # Returns
///
/// A 32-bit value assembled from the selected bytes.
///
/// # PTX
///
/// ```ptx
/// prmt.b32 result, a, b, c;
/// ```
///
/// # Example
///
/// ```rust,ignore
/// // Swap bytes 0 and 1 of a single word:
/// let swapped = prmt_b32(x, x, 0x0000_3210);
/// ```
#[inline(never)]
pub fn prmt_b32(a: u32, b: u32, c: u32) -> u32 {
    let _ = (a, b, c);
    unreachable!("prmt_b32 called outside CUDA kernel context")
}
