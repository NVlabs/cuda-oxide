/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use crate::verify::verify_operation;
use pliron::context::{Context, Ptr};
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Controls the reusable dialect-mir preparation stage.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MirPreparation {
    /// Promote stack slots to SSA and run annotation-driven loop unrolling.
    pub promote_and_unroll: bool,
    /// Print preparation-pass progress notes to stderr. Threaded from the
    /// pipeline's `BackendOptions`; the scalarization passes read this flag
    /// instead of the environment (loop unrolling still checks
    /// `CUDA_OXIDE_VERBOSE` on its own).
    pub verbose: bool,
}

/// Verify and prepare a dialect-mir module before LLVM lowering.
///
/// The one shared post-translation orchestrator calls this helper for both the
/// rustc and standalone frontends.
#[doc(hidden)]
pub fn prepare_mir_module(
    ctx: &mut Context,
    module: Ptr<Operation>,
    preparation: MirPreparation,
) -> Result<(), PipelineError> {
    verify_operation(ctx, module, "module")?;
    if !preparation.promote_and_unroll {
        return Ok(());
    }

    // A by-value aggregate argument initially lives in a MIR alloca. Read-only
    // field/index projections make that alloca non-promotable even though the
    // original entry-block argument is already an SSA value. Canonicalize the
    // validated pointer chains back to value extraction before mem2reg.
    mir_transforms::scalarize_borrowed_aggregate_reads::canonicalize_read_only_aggregate_arguments(
        module,
        ctx,
        preparation.verbose,
    );
    verify_operation(
        ctx,
        module,
        "module post-borrowed-aggregate-read-canonicalization",
    )?;

    let mut analyses = pliron::pass::AnalysisManager::default();
    pliron::opts::mem2reg::mem2reg(module, ctx, &mut analyses).map_err(|error| {
        PipelineError::Verification {
            name: "mem2reg".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        }
    })?;
    verify_operation(ctx, module, "module post-mem2reg")?;

    // An immutable aggregate pointer argument in an always-inline helper can
    // still retain dynamic field/array pointer chains after mem2reg. Recover
    // bounded read-only accesses in typed MIR before LLVM lowering.
    mir_transforms::scalarize_borrowed_aggregate_reads::
        canonicalize_bounded_borrowed_pointer_arguments(module, ctx, preparation.verbose);
    verify_operation(
        ctx,
        module,
        "module post-borrowed-pointer-read-canonicalization",
    )?;

    mir_transforms::unroll::unroll_annotated_loops(module, ctx, &mut analyses).map_err(
        |error| PipelineError::Verification {
            name: "loop-unroll".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        },
    )?;
    verify_operation(ctx, module, "module post-unroll")
}
