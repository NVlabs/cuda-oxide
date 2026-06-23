/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Annotation-driven loop unrolling over `dialect-mir`.
//!
//! Driven by a `mir.unroll` attribute (`UnrollAttr`) on a function op:
//! `0` = full unroll of a constant-trip-count loop, `n` = unroll by `n` with a
//! remainder loop. Built on pliron's IR cloning (`pliron::irbuild::cloning`).
//!
//! Loop validation is deliberately lightweight: per pliron-author guidance we
//! do not build a full `LoopInfo`. For an annotated function we locate the
//! natural loop's back-edge (a CFG edge classified `Back`) and from it the
//! header, latch, body, and exit, then unroll. LLVM's `LoopUnroll` is the
//! structural reference for the clone/rewire/remainder mechanics.

use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::result::Result;

/// Run annotation-driven loop unrolling over every function in `module` that
/// carries a `mir.unroll` attribute.
///
/// A no-op for functions without the attribute, so emitted IR is identical to
/// today unless `#[unroll]` / `#[unroll(N)]` is present.
pub fn unroll_annotated_loops(_module: Ptr<Operation>, _ctx: &mut Context) -> Result<()> {
    // Built incrementally:
    //   1. find each function op carrying `UnrollAttr`,
    //   2. locate the loop (back-edge via DFS edge classification + body),
    //   3. clone-and-rewire the body (full unroll, or partial-by-N + remainder)
    //      using `pliron::irbuild::cloning`.
    Ok(())
}
