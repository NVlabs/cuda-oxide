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

    let mut analyses = pliron::pass_manager::AnalysisManager::default();
    pliron::opts::mem2reg::mem2reg(module, ctx, &mut analyses).map_err(|error| {
        PipelineError::Verification {
            name: "mem2reg".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        }
    })?;
    verify_operation(ctx, module, "module post-mem2reg")?;

    mir_transforms::unroll::unroll_annotated_loops(module, ctx, &mut analyses).map_err(
        |error| PipelineError::Verification {
            name: "loop-unroll".to_string(),
            message: error.disp(ctx).to_string(),
            operation: None,
        },
    )?;
    verify_operation(ctx, module, "module post-unroll")
}
