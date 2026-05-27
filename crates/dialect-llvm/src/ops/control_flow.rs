/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Control flow operations - branches, switches, and terminators.
//!
//! This module contains LLVM dialect operations for control flow:
//!
//! ```text
//! ┌───────────────┬─────────────┬────────────────────────────────────────┐
//! │ Operation     │ LLVM Opcode │ Description                            │
//! ├───────────────┼─────────────┼────────────────────────────────────────┤
//! │ BrOp          │ br          │ Unconditional branch                   │
//! │ CondBrOp      │ br          │ Conditional branch (two successors)    │
//! │ SwitchOp      │ switch      │ Multi-way branch based on integer value│
//! │ ReturnOp      │ ret         │ Return from function                   │
//! │ UnreachableOp │ unreachable │ Mark unreachable code                  │
//! └───────────────┴─────────────┴────────────────────────────────────────┘
//! ```
//!
//! All branch operations implement [`BranchOpInterface`] for accessing successor operands.

use std::vec;

use combine::{
    between,
    parser::{Parser, char::spaces},
    token,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            self, BranchOpInterface, IsTerminatorInterface, NOpdsInterface, NResultsInterface,
            OperandSegmentInterface,
        },
        types::IntegerType,
    },
    common_traits::{Named, Verify},
    context::{Context, Ptr},
    derive::{op_interface_impl, pliron_op},
    identifier::Identifier,
    indented_block, input_err,
    irfmt::{
        self,
        parsers::{
            block_opd_parser, delimited_list_parser, process_parsed_ssa_defs, spaced,
            ssa_opd_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    location::{Located, Location},
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable, indented_nl},
    result::Result,
    r#type::Typed as TypedTrait,
    value::Value,
    verify_err,
};

use crate::{attributes::CaseValuesAttr, types::VoidType};

use super::symbol::FuncOp;

// ============================================================================
// Return Operation
// ============================================================================

/// Return from function.
///
/// Returns control flow back to the caller, optionally with a value.
///
/// Equivalent to LLVM's `ret` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description          |
/// |---------|----------------------|
/// | `arg`   | any type (optional)  |
/// ```
#[pliron_op(
    name = "llvm.return",
    format = "operands(CharSpace(`,`))",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>]
)]
pub struct ReturnOp;

impl ReturnOp {
    /// Create a new [`ReturnOp`].
    pub fn new(ctx: &mut Context, value: Option<Value>) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            value.into_iter().collect(),
            vec![],
            0,
        );
        ReturnOp { op }
    }

    /// Get the returned value, if it exists.
    #[must_use]
    pub fn retval(&self, ctx: &Context) -> Option<Value> {
        let op = &*self.get_operation().deref(ctx);
        if op.get_num_operands() == 1 {
            Some(op.get_operand(0))
        } else {
            None
        }
    }
}

impl Verify for ReturnOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = self.get_operation().deref(ctx);

        // If attached to a function, verify return type matches function signature
        if let Some(parent_op) = op.get_parent_op(ctx)
            && let Some(func_op) = Operation::get_op::<FuncOp>(parent_op, ctx)
        {
            let func_ty_ptr = func_op.get_type(ctx);
            let func_ty = func_ty_ptr.deref(ctx);
            let res_ty = func_ty.result_type();

            if res_ty.deref(ctx).is::<VoidType>() {
                if op.get_num_operands() != 0 {
                    return verify_err!(loc, "ReturnOp must have 0 operands for void function");
                }
            } else {
                if op.get_num_operands() != 1 {
                    return verify_err!(loc, "ReturnOp must have 1 operand for non-void function");
                }
                let ret_val = op.get_operand(0);
                let ret_ty = TypedTrait::get_type(&ret_val, ctx);
                if ret_ty != res_ty {
                    return verify_err!(
                        loc,
                        "ReturnOp type mismatch: expected {}, got {}",
                        res_ty.deref(ctx).disp(ctx),
                        ret_ty.deref(ctx).disp(ctx)
                    );
                }
            }
        }
        Ok(())
    }
}

// ============================================================================
// Unreachable Operation
// ============================================================================

/// Mark code as unreachable.
///
/// Indicates to LLVM that control flow cannot reach this point.
///
/// Equivalent to LLVM's `unreachable` instruction.
#[pliron_op(
    name = "llvm.unreachable",
    format = "",
    interfaces = [IsTerminatorInterface, NOpdsInterface<0>, NResultsInterface<0>]
)]
pub struct UnreachableOp;

impl UnreachableOp {
    /// Create a new [`UnreachableOp`].
    pub fn new(ctx: &mut Context) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        UnreachableOp { op }
    }
}

impl Verify for UnreachableOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = self.get_operation().deref(ctx);
        if op.get_num_successors() != 0 {
            return verify_err!(loc, "UnreachableOp must have 0 successors");
        }
        Ok(())
    }
}

// ============================================================================
// Unconditional Branch
// ============================================================================

/// Unconditional branch to a single successor.
///
/// Equivalent to LLVM's unconditional `br` instruction.
///
/// ### Operands
///
/// ```text
/// | operand     | description                              |
/// |-------------|------------------------------------------|
/// | `dest_opds` | Any number of operands with any LLVM type|
/// ```
///
/// ### Successors:
///
/// ```text
/// | Successor | description   |
/// |-----------|---------------|
/// | `dest`    | Any successor |
/// ```
#[pliron_op(
    name = "llvm.br",
    format = "succ($0) `(` operands(CharSpace(`,`)) `)`",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>]
)]
pub struct BrOp;

impl Verify for BrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = self.get_operation().deref(ctx);
        if op.get_num_successors() != 1 {
            return verify_err!(loc, "BrOp must have exactly one successor");
        }
        let succ = op.get_successor(0);
        let succ_num_args = succ.deref(ctx).get_num_arguments();
        let succ_opds = self.successor_operands(ctx, 0);

        if succ_num_args != succ_opds.len() {
            return verify_err!(loc, "BrOp operand count mismatch with successor arguments");
        }

        for (i, opd) in succ_opds.iter().enumerate() {
            let arg = succ.deref(ctx).get_argument(i);
            let arg_ty = TypedTrait::get_type(&arg, ctx);
            let opd_ty = TypedTrait::get_type(opd, ctx);
            if arg_ty != opd_ty {
                return verify_err!(
                    loc,
                    "BrOp operand {} type mismatch: expected {}, got {}",
                    i,
                    arg_ty.deref(ctx).disp(ctx),
                    opd_ty.deref(ctx).disp(ctx)
                );
            }
        }
        Ok(())
    }
}

#[op_interface_impl]
impl BranchOpInterface for BrOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        self.get_operation().deref(ctx).operands().collect()
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::push_operand(self.get_operation(), ctx, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        assert!(succ_idx == 0, "BrOp has exactly one successor");
        Operation::remove_operand(self.get_operation(), ctx, opd_idx)
    }
}

impl BrOp {
    /// Create a new [`BrOp`].
    pub fn new(ctx: &mut Context, dest: Ptr<BasicBlock>, dest_opds: Vec<Value>) -> Self {
        BrOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                dest_opds,
                vec![dest],
                0,
            ),
        }
    }
}

// ============================================================================
// Conditional Branch
// ============================================================================

/// Conditional branch with two successors.
///
/// Branches to `true_dest` if condition is true, `false_dest` otherwise.
///
/// Equivalent to LLVM's conditional `br` instruction.
///
/// ### Operands
///
/// ```text
/// | operand            | description                               |
/// |--------------------|-------------------------------------------|
/// | `condition`        | 1-bit signless integer                    |
/// | `true_dest_opds`   | Any number of operands with any LLVM type |
/// | `false_dest_opds`  | Any number of operands with any LLVM type |
/// ```
///
/// ### Successors:
///
/// ```text
/// | Successor    | description   |
/// |--------------|---------------|
/// | `true_dest`  | Any successor |
/// | `false_dest` | Any successor |
/// ```
#[pliron_op(
    name = "llvm.cond_br",
    interfaces = [IsTerminatorInterface, NResultsInterface<0>]
)]
pub struct CondBrOp;

impl CondBrOp {
    /// Create a new [`CondBrOp`].
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        true_dest: Ptr<BasicBlock>,
        true_dest_opds: Vec<Value>,
        false_dest: Ptr<BasicBlock>,
        false_dest_opds: Vec<Value>,
    ) -> Self {
        let (operands, segment_sizes) =
            Self::compute_segment_sizes(vec![vec![condition], true_dest_opds, false_dest_opds]);

        let op = CondBrOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                vec![true_dest, false_dest],
                0,
            ),
        };

        op.set_operand_segment_sizes(ctx, segment_sizes);
        op
    }

    /// Get the condition value for the branch.
    #[must_use]
    pub fn condition(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_operand(0)
    }
}

#[op_interface_impl]
impl OperandSegmentInterface for CondBrOp {}

impl Printable for CondBrOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        let condition = op.get_operand(0);
        let true_dest_opds = self.successor_operands(ctx, 0);
        let false_dest_opds = self.successor_operands(ctx, 1);
        write!(
            f,
            "{} if {} ^{}({}) else ^{}({})",
            Operation::get_opid(self.get_operation(), ctx),
            condition.disp(ctx),
            op.get_successor(0).deref(ctx).unique_name(ctx),
            iter_with_sep(
                true_dest_opds.iter(),
                printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
            op.get_successor(1).deref(ctx).unique_name(ctx),
            iter_with_sep(
                false_dest_opds.iter(),
                printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
        )
    }
}

impl Parsable for CondBrOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?;
        }

        let r#if = irfmt::parsers::spaced::<StateStream, _>(combine::parser::char::string("if"));
        let condition = ssa_opd_parser();
        let true_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let r_else =
            irfmt::parsers::spaced::<StateStream, _>(combine::parser::char::string("else"));
        let false_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());

        let final_parser = r#if
            .with(spaced(condition))
            .and(spaced(block_opd_parser()))
            .and(true_operands)
            .and(spaced(r_else).with(spaced(block_opd_parser()).and(false_operands)));

        final_parser
            .then(
                move |(((condition, true_dest), true_dest_opds), (false_dest, false_dest_opds))| {
                    let results = results.clone();
                    combine::parser(move |parsable_state: &mut StateStream<'a>| {
                        let ctx = &mut parsable_state.state.ctx;
                        let op = CondBrOp::new(
                            ctx,
                            condition,
                            true_dest,
                            true_dest_opds.clone(),
                            false_dest,
                            false_dest_opds.clone(),
                        );

                        process_parsed_ssa_defs(parsable_state, &results, op.get_operation())?;
                        Ok(OpObj::new(op)).into_parse_result()
                    })
                },
            )
            .parse_stream(state_stream)
            .into()
    }
}

impl Verify for CondBrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = self.get_operation().deref(ctx);
        if op.get_num_successors() != 2 {
            return verify_err!(loc, "CondBrOp must have exactly two successors");
        }

        // Verify condition is i1
        let cond = self.condition(ctx);
        let cond_ty = TypedTrait::get_type(&cond, ctx);
        if let Some(int_ty) = cond_ty.deref(ctx).downcast_ref::<IntegerType>() {
            if int_ty.width() != 1 {
                return verify_err!(loc, "CondBrOp condition must be i1");
            }
        } else {
            return verify_err!(loc, "CondBrOp condition must be integer type");
        }

        // Verify successors
        for i in 0..2 {
            let succ = op.get_successor(i);
            let succ_num_args = succ.deref(ctx).get_num_arguments();
            let succ_opds = self.successor_operands(ctx, i);

            if succ_num_args != succ_opds.len() {
                return verify_err!(loc, "CondBrOp successor {} operand count mismatch", i);
            }

            for (j, opd) in succ_opds.iter().enumerate() {
                let arg = succ.deref(ctx).get_argument(j);
                let arg_ty = TypedTrait::get_type(&arg, ctx);
                let opd_ty = TypedTrait::get_type(opd, ctx);
                if arg_ty != opd_ty {
                    return verify_err!(
                        loc,
                        "CondBrOp successor {} operand {} type mismatch: expected {}, got {}",
                        i,
                        j,
                        arg_ty.deref(ctx).disp(ctx),
                        opd_ty.deref(ctx).disp(ctx)
                    );
                }
            }
        }
        Ok(())
    }
}

#[op_interface_impl]
impl BranchOpInterface for CondBrOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        assert!(
            succ_idx == 0 || succ_idx == 1,
            "CondBrOp has exactly two successors"
        );
        // Skip the first segment, which is the condition.
        self.get_segment(ctx, succ_idx + 1)
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

// ============================================================================
// Switch Operation
// ============================================================================

/// One case of a switch statement.
#[derive(Clone)]
pub struct SwitchCase {
    /// The value being matched against.
    pub value: IntegerAttr,
    /// The destination block to jump to if this case is taken.
    pub dest: Ptr<BasicBlock>,
    /// The operands to pass to the destination block.
    pub dest_opds: Vec<Value>,
}

impl Printable for SwitchCase {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "{{ {}: ^{}({}) }}",
            self.value.disp(ctx),
            self.dest.deref(ctx).unique_name(ctx),
            list_with_sep(&self.dest_opds, printable::ListSeparator::CharSpace(',')).disp(ctx)
        )
    }
}

impl Parsable for SwitchCase {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        _arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        let mut parser = between(
            token('{'),
            token('}'),
            (
                spaced(IntegerAttr::parser(())),
                spaced(token(':')),
                spaced(block_opd_parser()),
                delimited_list_parser('(', ')', ',', ssa_opd_parser()),
                spaces(),
            ),
        );

        let ((value, _colon, dest, dest_opds, _spaces), _) =
            parser.parse_stream(state_stream).into_result()?;

        Ok(SwitchCase {
            value,
            dest,
            dest_opds,
        })
        .into_parse_result()
    }
}

/// Verification errors for [`SwitchOp`].
#[derive(thiserror::Error, Debug)]
pub enum SwitchOpVerifyErr {
    #[error("SwitchOp has no or incorrect case values attribute")]
    CaseValuesAttrErr,
    #[error("SwitchOp has no or incorrect default destination")]
    DefaultDestErr,
    #[error("SwitchOp has no condition operand or is not an integer")]
    ConditionErr,
}

/// Multi-way branch based on integer value.
///
/// Branches to different destinations based on comparing the condition against
/// case values. Has a default destination for unmatched values.
///
/// Equivalent to LLVM's `switch` instruction.
///
/// ### Operands
///
/// ```text
/// | operand             | description           |
/// |---------------------|-----------------------|
/// | `condition`         | integer type          |
/// | `default_dest_opds` | variadic of any type  |
/// | `case_dest_opds`    | variadic of any type  |
/// ```
///
/// ### Successors:
///
/// ```text
/// | Successor      | description       |
/// |----------------|-------------------|
/// | `default_dest` | any successor     |
/// | `case_dests`   | any successor(s)  |
/// ```
#[pliron_op(
    name = "llvm.switch",
    interfaces = [NResultsInterface<0>],
    attributes = (switch_case_values: CaseValuesAttr)
)]
pub struct SwitchOp;

impl Printable for SwitchOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let op = self.get_operation().deref(ctx);
        let condition = op.get_operand(0);

        let default_successor = op
            .successors()
            .next()
            .expect("SwitchOp must have at least one successor");
        let num_total_successors = op.get_num_successors();

        write!(
            f,
            "{} {}, ^{}({})",
            Operation::get_opid(self.get_operation(), ctx),
            condition.disp(ctx),
            default_successor.unique_name(ctx).disp(ctx),
            iter_with_sep(
                self.successor_operands(ctx, 0).iter(),
                printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
        )?;

        if num_total_successors < 2 {
            writeln!(f, "[]")?;
            return Ok(());
        }

        let cases = self.cases(ctx);

        write!(f, "{}[", indented_nl(state))?;
        indented_block!(state, {
            write!(f, "{}", indented_nl(state))?;
            list_with_sep(&cases, printable::ListSeparator::CharNewline(',')).fmt(ctx, state, f)?;
        });
        write!(f, "{}]", indented_nl(state))?;

        Ok(())
    }
}

impl Parsable for SwitchOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        arg: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !arg.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, arg.len())
            )?;
        }

        let condition = ssa_opd_parser().skip(spaced(token(',')));
        let default_successor = block_opd_parser();
        let default_operands = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let cases = delimited_list_parser('[', ']', ',', SwitchCase::parser(()));

        let final_parser = spaced(condition)
            .and(default_successor)
            .skip(spaces())
            .and(default_operands)
            .skip(spaces())
            .and(cases);

        final_parser
            .then(
                move |(((condition, default_dest), default_dest_opds), cases)| {
                    let results = arg.clone();
                    combine::parser(move |parsable_state: &mut StateStream<'a>| {
                        let ctx = &mut parsable_state.state.ctx;
                        let op = SwitchOp::new(
                            ctx,
                            condition,
                            default_dest,
                            default_dest_opds.clone(),
                            cases.clone(),
                        );

                        process_parsed_ssa_defs(parsable_state, &results, op.get_operation())?;
                        Ok(OpObj::new(op)).into_parse_result()
                    })
                },
            )
            .parse_stream(state_stream)
            .into()
    }
}

impl SwitchOp {
    /// Create a new [`SwitchOp`].
    pub fn new(
        ctx: &mut Context,
        condition: Value,
        default_dest: Ptr<BasicBlock>,
        default_dest_opds: Vec<Value>,
        cases: Vec<SwitchCase>,
    ) -> Self {
        let case_values: Vec<IntegerAttr> = cases.iter().map(|case| case.value.clone()).collect();

        let case_operands = cases
            .iter()
            .map(|case| case.dest_opds.clone())
            .collect::<Vec<_>>();

        let mut operand_segments = vec![vec![condition], default_dest_opds];
        operand_segments.extend(case_operands);
        let (operands, segment_sizes) = Self::compute_segment_sizes(operand_segments);

        let case_dests = cases.iter().map(|case| case.dest);
        let successors = vec![default_dest].into_iter().chain(case_dests).collect();
        let op = SwitchOp {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                operands,
                successors,
                0,
            ),
        };

        op.set_operand_segment_sizes(ctx, segment_sizes);
        op.set_attr_switch_case_values(ctx, CaseValuesAttr(case_values));
        op
    }

    /// Get the cases of this switch operation.
    /// (The default case cannot be / isn't included here).
    #[must_use]
    pub fn cases(&self, ctx: &Context) -> Vec<SwitchCase> {
        let case_values = &*self
            .get_attr_switch_case_values(ctx)
            .expect("SwitchOp missing or incorrect case values attribute");

        let op = self.get_operation().deref(ctx);
        // Skip the first one, which is the default successor.
        let successors = op.successors().skip(1);

        successors
            .zip(case_values.0.iter())
            .enumerate()
            .map(|(i, (dest, value))| {
                // i+1 here because the first successor is the default destination.
                let dest_opds = self.successor_operands(ctx, i + 1);
                SwitchCase {
                    value: value.clone(),
                    dest,
                    dest_opds,
                }
            })
            .collect()
    }

    /// Get the condition value for the switch.
    #[must_use]
    pub fn condition(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the default destination of this switch operation.
    #[must_use]
    pub fn default_dest(&self, ctx: &Context) -> Ptr<BasicBlock> {
        self.get_operation().deref(ctx).get_successor(0)
    }
}

#[op_interface_impl]
impl IsTerminatorInterface for SwitchOp {}

#[op_interface_impl]
impl BranchOpInterface for SwitchOp {
    fn successor_operands(&self, ctx: &Context, succ_idx: usize) -> Vec<Value> {
        // Skip the first segment, which is the condition.
        self.get_segment(ctx, succ_idx + 1)
    }

    fn add_successor_operand(&self, ctx: &mut Context, succ_idx: usize, operand: Value) -> usize {
        self.push_to_segment(ctx, succ_idx + 1, operand)
    }

    fn remove_successor_operand(
        &self,
        ctx: &mut Context,
        succ_idx: usize,
        opd_idx: usize,
    ) -> Value {
        self.remove_from_segment(ctx, succ_idx + 1, opd_idx)
    }
}

#[op_interface_impl]
impl OperandSegmentInterface for SwitchOp {}

impl Verify for SwitchOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::r#type::TypePtr;

        let loc = self.loc(ctx);

        let Some(case_values) = self.get_attr_switch_case_values(ctx) else {
            verify_err!(loc.clone(), SwitchOpVerifyErr::CaseValuesAttrErr)?
        };

        let op = &*self.get_operation().deref(ctx);

        if op.get_num_successors() < 1 {
            verify_err!(loc.clone(), SwitchOpVerifyErr::DefaultDestErr)?;
        }

        if op.get_num_operands() < 1 {
            verify_err!(loc.clone(), SwitchOpVerifyErr::ConditionErr)?;
        }

        let condition_ty = pliron::r#type::Typed::get_type(&op.get_operand(0), ctx);
        let condition_ty = TypePtr::<IntegerType>::from_ptr(condition_ty, ctx)?;

        if let Some(case_value) = case_values.0.first() {
            // Ensure that the case value type matches the condition type.
            if case_value.get_type() != condition_ty {
                verify_err!(loc, SwitchOpVerifyErr::ConditionErr)?;
            }
        }

        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all control flow operations.
pub fn register(ctx: &mut Context) {
    ReturnOp::register(ctx);
    UnreachableOp::register(ctx);
    BrOp::register(ctx);
    CondBrOp::register(ctx);
    SwitchOp::register(ctx);
}
