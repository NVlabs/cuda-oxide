/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::error::PipelineError;
use pliron::common_traits::Verify;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::operation::Operation;
use pliron::printable::Printable;

/// Recursively verifies an operation and all nested operations.
///
/// On failure, attempts to find the innermost failing operation for better
/// error messages.
// mir-importer pipeline plumbing; not part of the frontend contract.
#[doc(hidden)]
pub fn verify_operation(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
    name: &str,
) -> Result<(), PipelineError> {
    if let Err(e) = op_ptr.deref(ctx).verify(ctx) {
        // Try to find specific failing operation
        if let Some((err_op, err_msg)) = find_inner_verification_error(ctx, op_ptr) {
            return Err(PipelineError::Verification {
                name: name.to_string(),
                message: err_msg,
                operation: Some(err_op.deref(ctx).disp(ctx).to_string()),
            });
        }

        // Use .disp(ctx) to get full error with location and backtrace
        return Err(PipelineError::Verification {
            name: name.to_string(),
            message: e.disp(ctx).to_string(),
            operation: None,
        });
    }
    Ok(())
}

/// Recursively finds the innermost operation that failed verification.
///
/// Helps produce better error messages by pointing to the specific failing
/// operation rather than just the containing module/function.
fn find_inner_verification_error(
    ctx: &Context,
    op_ptr: Ptr<Operation>,
) -> Option<(Ptr<Operation>, String)> {
    let op = op_ptr.deref(ctx);

    for region in op.regions() {
        let region_ref = region.deref(ctx);
        for block in region_ref.iter(ctx) {
            let block_ref = block.deref(ctx);
            for child_op in block_ref.iter(ctx) {
                if let Some(err) = find_inner_verification_error(ctx, child_op) {
                    return Some(err);
                }
            }
        }
    }

    if let Err(e) = op.verify(ctx) {
        // Use .disp(ctx) to get full error with location and backtrace
        return Some((op_ptr, e.disp(ctx).to_string()));
    }

    None
}
