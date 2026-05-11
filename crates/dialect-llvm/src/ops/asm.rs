/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Inline assembly operations.
//!
//! This module contains LLVM dialect operations for inline assembly:
//!
//! ```text
//! ┌──────────────────┬───────────────────────────────────────────────────┐
//! │ Operation        │ Description                                       │
//! ├──────────────────┼───────────────────────────────────────────────────┤
//! │ InlineAsmOp      │ Single-result inline assembly                     │
//! │ InlineAsmMultiOp │ Multi-result inline asm with tied operand support │
//! └──────────────────┴───────────────────────────────────────────────────┘
//! ```
//!
//! # Inline Assembly
//!
//! Inline assembly allows embedding target-specific assembly instructions
//! directly in LLVM IR. This is essential for GPU programming where certain
//! instructions (like warp shuffles, barriers, TMA operations) have no
//! high-level equivalent.
//!
//! ## Constraints
//!
//! Constraints specify how operands are passed to/from assembly:
//! - `=r` - output to general register
//! - `=f` - output to float register
//! - `=l` - output to 64-bit register
//! - `r` - input from general register
//! - `f` - input from float register
//! - `l` - input from 64-bit register
//! - `0`, `1`, etc. - tied to output operand N
//!
//! ## Tied Operands
//!
//! Tied operands share the same register for input and output, essential for
//! read-modify-write operations like WGMMA accumulators.

use pliron::{
    builtin::attributes::{BoolAttr, StringAttr},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_op, derive_attr_get_set},
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    r#type::TypeObj,
    value::Value,
    verify_err,
};

use crate::types::VoidType;

// ============================================================================
// InlineAsm Operation
// ============================================================================

/// Single-result inline assembly operation.
///
/// Represents LLVM inline assembly:
/// ```llvm
/// %result = call <ret_ty> asm sideeffect "<asm_string>", "<constraints>"(<args>...)
/// ```
///
/// ### Operands
///
/// ```text
/// | operand | description |
/// |---------|-------------|
/// | `inputs` | Input operands to the inline asm |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description |
/// |--------|-------------|
/// | `result` | Output value (optional, void if no output) |
/// ```
///
/// ### Attributes:
/// - `asm_template`: The assembly template string
/// - `constraints`: The constraint string (e.g., "=l,r,r")
/// - `convergent`: Whether the asm requires warp-synchronous execution
#[def_op("llvm.inline_asm")]
#[pliron::derive::format_op("operands(CharSpace(`,`))")]
#[derive_attr_get_set(
    inline_asm_template: StringAttr,
    inline_asm_constraints: StringAttr,
    inline_asm_convergent: BoolAttr
)]
pub struct InlineAsmOp;

impl InlineAsmOp {
    /// Create a new `InlineAsmOp` with a result.
    ///
    /// # Arguments
    /// * `ctx` - The Pliron IR context
    /// * `result_ty` - The result type (use `VoidType` for no result)
    /// * `inputs` - Input operands
    /// * `asm_template` - The assembly template string
    /// * `constraints` - The constraint string
    pub fn new(
        ctx: &mut Context,
        result_ty: Ptr<TypeObj>,
        inputs: Vec<Value>,
        asm_template: &str,
        constraints: &str,
    ) -> Self {
        let is_void = result_ty.deref(ctx).is::<VoidType>();
        let results = if is_void { vec![] } else { vec![result_ty] };

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            results,
            inputs,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_inline_asm_template(ctx, StringAttr::new(asm_template.into()));
        op.set_attr_inline_asm_constraints(ctx, StringAttr::new(constraints.into()));
        op
    }

    /// Create a new convergent `InlineAsmOp`.
    ///
    /// Use this for operations that require warp-synchronous execution semantics,
    /// such as barriers, mbarrier operations, and warp shuffles.
    /// The convergent attribute tells LLVM not to apply optimizations that would
    /// change which threads execute this operation together.
    pub fn new_convergent(
        ctx: &mut Context,
        result_ty: Ptr<TypeObj>,
        inputs: Vec<Value>,
        asm_template: &str,
        constraints: &str,
    ) -> Self {
        let op = Self::new(ctx, result_ty, inputs, asm_template, constraints);
        op.set_attr_inline_asm_convergent(ctx, BoolAttr::new(true));
        op
    }

    /// Get the assembly template string.
    #[must_use]
    pub fn asm_template(&self, ctx: &Context) -> String {
        self.get_attr_inline_asm_template(ctx)
            .map(|attr| String::from(attr.clone()))
            .unwrap_or_default()
    }

    /// Get the constraint string.
    #[must_use]
    pub fn constraints(&self, ctx: &Context) -> String {
        self.get_attr_inline_asm_constraints(ctx)
            .map(|attr| String::from(attr.clone()))
            .unwrap_or_default()
    }

    /// Check if this inline asm is marked as convergent.
    #[must_use]
    pub fn is_convergent(&self, ctx: &Context) -> bool {
        self.get_attr_inline_asm_convergent(ctx)
            .is_some_and(|attr| bool::from(attr.clone()))
    }
}

impl Verify for InlineAsmOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_attr_inline_asm_template(ctx).is_none() {
            return verify_err!(
                self.get_operation().deref(ctx).loc(),
                "llvm.inline_asm requires 'inline_asm_template' attribute"
            );
        }
        if self.get_attr_inline_asm_constraints(ctx).is_none() {
            return verify_err!(
                self.get_operation().deref(ctx).loc(),
                "llvm.inline_asm requires 'inline_asm_constraints' attribute"
            );
        }
        Ok(())
    }
}

// ============================================================================
// InlineAsmMulti Operation
// ============================================================================

/// Multi-result inline assembly operation with tied operand support.
///
/// This is a more powerful version of `InlineAsmOp` that supports:
/// - Multiple output values (not just 0 or 1)
/// - Tied operands (input tied to output, using the same register)
/// - Full constraint specification
///
/// ## Use Cases
/// - WGMMA instructions with 32 accumulator registers
/// - User-provided inline PTX in kernel code
/// - Complex multi-output assembly sequences
///
/// ## Tied Operands
///
/// Tied operands mean an input and output share the same register.
/// In GCC syntax: `"+r"(x)` or `"=r"(x) : "0"(x)`
///
/// The `tied_inputs` attribute is a comma-separated string where each entry
/// corresponds to an input operand:
/// - `-1`: Not tied to any output
/// - `N`: Tied to output operand N (0-indexed)
///
/// ## LLVM IR Output
///
/// Multi-output inline asm returns a struct in LLVM:
/// ```llvm
/// %result = call {f32, f32, ...} asm sideeffect "...", "=f,=f,...,0,1,...,l,l"(...)
/// %out0 = extractvalue {f32, f32, ...} %result, 0
/// %out1 = extractvalue {f32, f32, ...} %result, 1
/// ```
#[def_op("llvm.inline_asm_multi")]
#[pliron::derive::format_op("operands(CharSpace(`,`))")]
#[derive_attr_get_set(
    multi_asm_template: StringAttr,
    multi_asm_constraints: StringAttr,
    multi_asm_tied_inputs: StringAttr,
    multi_asm_convergent: BoolAttr
)]
pub struct InlineAsmMultiOp;

impl InlineAsmMultiOp {
    /// Create a new multi-result inline asm operation.
    ///
    /// # Arguments
    /// * `ctx` - The Pliron IR context
    /// * `result_types` - Vector of result types (can be empty for void)
    /// * `inputs` - Input operands
    /// * `asm_template` - The assembly template string
    /// * `constraints` - The full constraint string
    /// * `tied_inputs` - For each input, which output it's tied to (-1 if not tied)
    pub fn new(
        ctx: &mut Context,
        result_types: Vec<Ptr<TypeObj>>,
        inputs: Vec<Value>,
        asm_template: &str,
        constraints: &str,
        tied_inputs: &[i32],
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            inputs,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_multi_asm_template(ctx, StringAttr::new(asm_template.into()));
        op.set_attr_multi_asm_constraints(ctx, StringAttr::new(constraints.into()));

        // Convert tied_inputs to comma-separated string
        let tied_str = tied_inputs
            .iter()
            .map(std::string::ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        op.set_attr_multi_asm_tied_inputs(ctx, StringAttr::new(tied_str));

        op
    }

    /// Create a new convergent multi-result inline asm operation.
    ///
    /// Use this for operations requiring warp-synchronous execution (WGMMA, barriers, etc.)
    pub fn new_convergent(
        ctx: &mut Context,
        result_types: Vec<Ptr<TypeObj>>,
        inputs: Vec<Value>,
        asm_template: &str,
        constraints: &str,
        tied_inputs: &[i32],
    ) -> Self {
        let op = Self::new(
            ctx,
            result_types,
            inputs,
            asm_template,
            constraints,
            tied_inputs,
        );
        op.set_attr_multi_asm_convergent(ctx, BoolAttr::new(true));
        op
    }

    /// Convenience constructor for tied read-write operands (like WGMMA accumulators).
    ///
    /// Creates an asm operation where the first N inputs are tied to N outputs,
    /// and the remaining inputs are untied.
    ///
    /// # Arguments
    /// * `ctx` - The Pliron IR context
    /// * `num_tied` - Number of tied input/output pairs
    /// * `output_ty` - Type for each tied output (e.g., f32)
    /// * `tied_inputs` - The tied input values (will be both input and output)
    /// * `other_inputs` - Additional non-tied inputs
    /// * `asm_template` - The assembly template string
    /// * `output_constraint` - Constraint for outputs (e.g., "f" for float reg)
    /// * `other_constraints` - Constraints for non-tied inputs (e.g., "l,l" for two i64)
    #[allow(clippy::too_many_arguments)]
    pub fn new_tied_convergent(
        ctx: &mut Context,
        num_tied: u32,
        output_ty: Ptr<TypeObj>,
        tied_inputs: Vec<Value>,
        other_inputs: Vec<Value>,
        asm_template: &str,
        output_constraint: &str,
        other_constraints: &str,
    ) -> Self {
        assert_eq!(tied_inputs.len(), num_tied as usize);

        // Build result types: num_tied outputs of output_ty
        let result_types: Vec<_> = (0..num_tied as usize).map(|_| output_ty).collect();

        // Build tied_inputs array: first num_tied are tied to 0..num_tied, rest are -1
        let num_other = other_inputs.len();
        #[allow(clippy::cast_possible_wrap)] // num_tied is always small (< 100)
        let mut tied: Vec<i32> = (0..num_tied as i32).collect();
        tied.extend(std::iter::repeat_n(-1, num_other));

        // Build inputs: tied inputs first, then other inputs
        let mut inputs = tied_inputs;
        inputs.extend(other_inputs);

        // Build constraint string:
        // "=f,=f,...,=f,  0,1,...,N-1,  l,l,..."
        //  ^outputs^     ^tied inputs^  ^other^
        let mut constraints = String::new();

        // Output constraints
        for i in 0..num_tied {
            if i > 0 {
                constraints.push(',');
            }
            constraints.push('=');
            constraints.push_str(output_constraint);
        }

        // Tied input constraints (reference output by number)
        for i in 0..num_tied {
            constraints.push(',');
            constraints.push_str(&i.to_string());
        }

        // Other input constraints
        if !other_constraints.is_empty() {
            constraints.push(',');
            constraints.push_str(other_constraints);
        }

        Self::new_convergent(ctx, result_types, inputs, asm_template, &constraints, &tied)
    }

    /// Get the assembly template string.
    #[must_use]
    pub fn asm_template(&self, ctx: &Context) -> String {
        self.get_attr_multi_asm_template(ctx)
            .map(|attr| String::from(attr.clone()))
            .unwrap_or_default()
    }

    /// Get the constraint string.
    #[must_use]
    pub fn constraints(&self, ctx: &Context) -> String {
        self.get_attr_multi_asm_constraints(ctx)
            .map(|attr| String::from(attr.clone()))
            .unwrap_or_default()
    }

    /// Get the tied inputs as a vector of indices.
    #[must_use]
    pub fn tied_inputs(&self, ctx: &Context) -> Vec<i32> {
        self.get_attr_multi_asm_tied_inputs(ctx)
            .map(|attr| {
                String::from(attr.clone())
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| s.parse::<i32>().unwrap_or(-1))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Check if this inline asm is marked as convergent.
    #[must_use]
    pub fn is_convergent(&self, ctx: &Context) -> bool {
        self.get_attr_multi_asm_convergent(ctx)
            .is_some_and(|attr| bool::from(attr.clone()))
    }

    /// Get the number of results.
    #[must_use]
    pub fn num_results(&self, ctx: &Context) -> usize {
        self.get_operation().deref(ctx).get_num_results()
    }
}

impl Verify for InlineAsmMultiOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_attr_multi_asm_template(ctx).is_none() {
            return verify_err!(
                self.get_operation().deref(ctx).loc(),
                "llvm.inline_asm_multi requires 'multi_asm_template' attribute"
            );
        }
        if self.get_attr_multi_asm_constraints(ctx).is_none() {
            return verify_err!(
                self.get_operation().deref(ctx).loc(),
                "llvm.inline_asm_multi requires 'multi_asm_constraints' attribute"
            );
        }
        if self.get_attr_multi_asm_tied_inputs(ctx).is_none() {
            return verify_err!(
                self.get_operation().deref(ctx).loc(),
                "llvm.inline_asm_multi requires 'multi_asm_tied_inputs' attribute"
            );
        }
        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all inline assembly operations.
pub fn register(ctx: &mut Context) {
    InlineAsmOp::register(ctx);
    InlineAsmMultiOp::register(ctx);
}
