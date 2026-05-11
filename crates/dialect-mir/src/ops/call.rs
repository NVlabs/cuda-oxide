/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR call operations.
//!
//! This module defines function call operations for the MIR dialect.

use pliron::{
    builtin::attributes::StringAttr,
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    verify_err,
};
use pliron_derive::pliron_op;

// ============================================================================
// MirCallOp
// ============================================================================

/// MIR call operation.
///
/// Represents a function call.
///
/// # Attributes
///
/// ```text
/// | Name     | Type       | Description                |
/// |----------|------------|----------------------------|
/// | `callee` | StringAttr | Name of the called function |
/// ```
///
/// # Operands
///
/// Variadic operands matching the callee's argument types.
///
/// # Results
///
/// Variadic results matching the callee's return types.
///
/// # Verification
///
/// - Must have `callee` attribute.
/// - Argument type verification is deferred to later passes that have
///   access to the symbol table.
#[pliron_op(
    name = "mir.call",
    format,
    attributes = (callee: StringAttr)
)]
pub struct MirCallOp;

impl MirCallOp {
    /// Create a new MirCallOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        Self { op }
    }
}

impl Verify for MirCallOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Check callee attribute exists
        if self.get_attr_callee(ctx).is_none() {
            return verify_err!(op.loc(), "MirCallOp must have a callee attribute");
        }

        // Note: We cannot easily verify argument types against the callee signature
        // without a symbol table lookup which might be expensive or complex here.
        // For now, we strictly verify the structure.

        Ok(())
    }
}

/// Register call operations into the given context.
pub fn register(ctx: &mut Context) {
    MirCallOp::register(ctx);
}
