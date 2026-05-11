/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function call operations.
//!
//! This module contains LLVM dialect operations for calling functions:
//!
//! ```text
//! ┌──────────────────┬────────────────────────────────────────┐
//! │ Operation        │ Description                            │
//! ├──────────────────┼────────────────────────────────────────┤
//! │ CallOp           │ Direct or indirect function call       │
//! │ CallIntrinsicOp  │ Call to LLVM intrinsic function        │
//! └──────────────────┴────────────────────────────────────────┘
//! ```

use combine::parser::Parser;
use pliron::{
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{IdentifierAttr, StringAttr, TypeAttr},
        op_interfaces::{
            CallOpCallable, CallOpInterface, NResultsInterface, OneResultInterface,
            SymbolUserOpInterface,
        },
        type_interfaces::FunctionTypeInterface,
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface_impl, pliron_op},
    identifier::Identifier,
    irfmt::{
        parsers::{delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser},
        printers::list_with_sep,
    },
    location::{Located, Location},
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable},
    result::Result,
    symbol_table::SymbolTableCollection,
    r#type::{TypePtr, Typed},
    value::Value,
    verify_err,
};

use crate::{
    attributes::FastmathFlagsAttr,
    types::{FuncType, PointerType},
};

use super::symbol::{FuncOp, SymbolUserOpVerifyErr};

// ============================================================================
// Call Operation
// ============================================================================

/// Direct or indirect function call.
///
/// Calls a function either directly by name or indirectly via a function pointer.
///
/// Equivalent to LLVM's `call` instruction.
///
/// ### Operands
///
/// ```text
/// | operand           | description                                               |
/// |-------------------|-----------------------------------------------------------|
/// | `callee_operands` | Optional function pointer followed by any number of params|
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description              |
/// |--------|--------------------------|
/// | `res`  | LLVM type (return value) |
/// ```
#[pliron_op(
    name = "llvm.call",
    interfaces = [NResultsInterface<1>, OneResultInterface],
    attributes = (llvm_call_callee: IdentifierAttr, llvm_call_fastmath_flags: FastmathFlagsAttr)
)]
pub struct CallOp;

impl CallOp {
    /// Create a new [`CallOp`].
    pub fn new(
        ctx: &mut Context,
        callee: CallOpCallable,
        callee_ty: TypePtr<FuncType>,
        mut args: Vec<Value>,
    ) -> Self {
        let res_ty = callee_ty.deref(ctx).result_type();
        let op = match callee {
            CallOpCallable::Direct(cval) => {
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![res_ty],
                    args,
                    vec![],
                    0,
                );
                let op = Self { op };
                op.set_attr_llvm_call_callee(ctx, IdentifierAttr::new(cval));
                op
            }
            CallOpCallable::Indirect(csym) => {
                args.insert(0, csym);
                let op = Operation::new(
                    ctx,
                    Self::get_concrete_op_info(),
                    vec![res_ty],
                    args,
                    vec![],
                    0,
                );
                Self { op }
            }
        };
        op.set_callee_type(ctx, callee_ty.into());
        op
    }
}

#[op_interface_impl]
impl SymbolUserOpInterface for CallOp {
    fn verify_symbol_uses(
        &self,
        ctx: &Context,
        symbol_tables: &mut SymbolTableCollection,
    ) -> Result<()> {
        match self.callee(ctx) {
            CallOpCallable::Direct(callee_sym) => {
                let Some(callee) = symbol_tables.lookup_symbol_in_nearest_table(
                    ctx,
                    self.get_operation(),
                    &callee_sym,
                ) else {
                    return verify_err!(
                        self.loc(ctx),
                        SymbolUserOpVerifyErr::SymbolNotFound(callee_sym.to_string())
                    );
                };
                let Some(func_op) = (&*callee as &dyn Op).downcast_ref::<FuncOp>() else {
                    return verify_err!(
                        self.loc(ctx),
                        SymbolUserOpVerifyErr::NotLlvmFunc(callee_sym.to_string())
                    );
                };
                let func_op_ty = func_op.get_type(ctx);

                if func_op_ty.to_ptr() != self.callee_type(ctx) {
                    return verify_err!(
                        self.loc(ctx),
                        SymbolUserOpVerifyErr::FuncTypeErr(format!(
                            "expected {}, got {}",
                            func_op_ty.disp(ctx),
                            self.callee_type(ctx).disp(ctx)
                        ))
                    );
                }
            }
            CallOpCallable::Indirect(pointer) => {
                if !pointer.get_type(ctx).deref(ctx).is::<PointerType>() {
                    return verify_err!(
                        self.loc(ctx),
                        SymbolUserOpVerifyErr::FuncTypeErr("Callee must be a pointer".to_string())
                    );
                }
            }
        }
        Ok(())
    }

    fn used_symbols(&self, ctx: &Context) -> Vec<Identifier> {
        match self.callee(ctx) {
            CallOpCallable::Direct(identifier) => vec![identifier],
            CallOpCallable::Indirect(_) => vec![],
        }
    }
}

#[op_interface_impl]
impl CallOpInterface for CallOp {
    fn callee(&self, ctx: &Context) -> CallOpCallable {
        let op = self.op.deref(ctx);
        if let Some(callee_sym) = self.get_attr_llvm_call_callee(ctx) {
            CallOpCallable::Direct(callee_sym.clone().into())
        } else {
            assert!(
                op.get_num_operands() > 0,
                "Indirect call must have function pointer operand"
            );
            CallOpCallable::Indirect(op.get_operand(0))
        }
    }

    fn args(&self, ctx: &Context) -> Vec<Value> {
        let op = self.op.deref(ctx);
        // If this is an indirect call, the first operand is the callee value.
        let skip = usize::from(!matches!(self.callee(ctx), CallOpCallable::Direct(_)));
        op.operands().skip(skip).collect()
    }
}

impl Printable for CallOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        let callee = self.callee(ctx);
        write!(
            f,
            "{} = {} ",
            self.get_result(ctx).disp(ctx),
            self.get_opid()
        )?;
        match callee {
            CallOpCallable::Direct(callee_sym) => {
                write!(f, "@{callee_sym}")?;
            }
            CallOpCallable::Indirect(callee_val) => {
                write!(f, "{}", callee_val.disp(ctx))?;
            }
        }

        if let Some(fmf) = self.get_attr_llvm_call_fastmath_flags(ctx)
            && *fmf != FastmathFlagsAttr::default()
        {
            write!(f, " {}", fmf.disp(ctx))?;
        }

        let args = self.args(ctx);
        let ty = self.callee_type(ctx);
        write!(
            f,
            " ({}) : {}",
            list_with_sep(&args, printable::ListSeparator::CharSpace(',')).disp(ctx),
            ty.disp(ctx)
        )?;
        Ok(())
    }
}

impl Parsable for CallOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::{optional, token};

        let direct_callee = token('@')
            .with(Identifier::parser(()))
            .map(CallOpCallable::Direct);
        let indirect_callee = ssa_opd_parser().map(CallOpCallable::Indirect);
        let callee_parser = direct_callee.or(indirect_callee);
        let fastmath_flags_parser = optional(FastmathFlagsAttr::parser(()));
        let args_parser = delimited_list_parser('(', ')', ',', ssa_opd_parser());
        let ty_parser = spaced(token(':')).with(TypePtr::<FuncType>::parser(()));

        let mut final_parser = spaced(callee_parser)
            .and(spaced(fastmath_flags_parser))
            .and(spaced(args_parser))
            .and(ty_parser)
            .then(move |(((callee, fastmath_flags), args), ty)| {
                let results = results.clone();
                combine::parser(move |parsable_state: &mut StateStream<'a>| {
                    let ctx = &mut parsable_state.state.ctx;
                    let op = Self::new(ctx, callee.clone(), ty, args.clone());
                    if let Some(fmf) = &fastmath_flags {
                        op.set_attr_llvm_call_fastmath_flags(ctx, *fmf);
                    }
                    process_parsed_ssa_defs(parsable_state, &results, op.get_operation())?;
                    Ok(OpObj::new(op)).into_parse_result()
                })
            });

        final_parser.parse_stream(state_stream).into()
    }
}

impl Verify for CallOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        // Check that the argument and result types match the callee type.
        let callee_ty = &*self.callee_type(ctx).deref(ctx);
        let Some(callee_ty) = callee_ty.downcast_ref::<FuncType>() else {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::FuncTypeErr("Callee is not a function".to_string())
            );
        };
        // Check the function type against the arguments.
        let args = self.args(ctx);
        let expected_args = callee_ty.arg_types();
        if !callee_ty.is_var_arg() && args.len() != expected_args.len() {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::FuncTypeErr("argument count mismatch.".to_string())
            );
        }
        for (arg_idx, (arg, expected_arg)) in args.iter().zip(expected_args.iter()).enumerate() {
            if arg.get_type(ctx) != *expected_arg {
                return verify_err!(
                    self.loc(ctx),
                    SymbolUserOpVerifyErr::FuncTypeErr(format!(
                        "argument {} type mismatch: expected {}, got {}",
                        arg_idx,
                        expected_arg.disp(ctx),
                        arg.get_type(ctx).disp(ctx)
                    ))
                );
            }
        }

        if callee_ty.result_type() != self.result_type(ctx) {
            return verify_err!(
                self.loc(ctx),
                SymbolUserOpVerifyErr::FuncTypeErr(format!(
                    "result type mismatch: expected {}, got {}",
                    callee_ty.result_type().disp(ctx),
                    self.result_type(ctx).disp(ctx)
                ))
            );
        }

        Ok(())
    }
}

// ============================================================================
// CallIntrinsic Operation
// ============================================================================

/// Verification errors for [`CallIntrinsicOp`].
#[derive(thiserror::Error, Debug)]
pub enum CallIntrinsicVerifyErr {
    #[error("Missing or incorrect intrinsic name attribute")]
    MissingIntrinsicNameAttr,
    #[error("Missing or incorrect intrinsic type attribute")]
    MissingIntrinsicTypeAttr,
    #[error("Number or types of operands does not match intrinsic type")]
    OperandsMismatch,
    #[error("Number or types of results does not match intrinsic type")]
    ResultsMismatch,
    #[error("Intrinsic name does not correspond to a known LLVM intrinsic")]
    UnknownIntrinsicName,
}

/// Call to an LLVM intrinsic function.
///
/// All LLVM intrinsic calls are represented by this operation.
///
/// Same as upstream MLIR's [llvm.call_intrinsic](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmcall_intrinsic-llvmcallintrinsicop).
#[pliron_op(
    name = "llvm.call_intrinsic",
    interfaces = [NResultsInterface<1>, OneResultInterface],
    attributes = (
        llvm_intrinsic_name: StringAttr,
        llvm_intrinsic_type: TypeAttr,
        llvm_intrinsic_fastmath_flags: FastmathFlagsAttr
    )
)]
pub struct CallIntrinsicOp;

impl CallIntrinsicOp {
    /// Create a new [`CallIntrinsicOp`].
    pub fn new(
        ctx: &mut Context,
        intrinsic_name: StringAttr,
        intrinsic_type: TypePtr<FuncType>,
        operands: Vec<Value>,
    ) -> Self {
        let res_ty = intrinsic_type.deref(ctx).result_type();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![res_ty],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_llvm_intrinsic_name(ctx, intrinsic_name);
        op.set_attr_llvm_intrinsic_type(ctx, TypeAttr::new(intrinsic_type.into()));
        op
    }
}

impl Printable for CallIntrinsicOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use pliron::irfmt::printers::iter_with_sep;

        // [result = ] llvm.call_intrinsic @name <FastMathFlags> (operands) : type
        if let Some(res) = self.op.deref(ctx).results().next() {
            write!(f, "{} = ", res.disp(ctx))?;
        }

        write!(
            f,
            "{} @{} ",
            Self::get_opid_static(),
            self.get_attr_llvm_intrinsic_name(ctx)
                .expect("CallIntrinsicOp missing or incorrect intrinsic name attribute")
                .disp(ctx),
        )?;

        if let Some(fmf) = self.get_attr_llvm_intrinsic_fastmath_flags(ctx)
            && *fmf != FastmathFlagsAttr::default()
        {
            write!(f, " {} ", fmf.disp(ctx))?;
        }

        write!(
            f,
            "({}) : {}",
            iter_with_sep(
                self.op.deref(ctx).operands(),
                printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
            self.get_attr_llvm_intrinsic_type(ctx)
                .expect("CallIntrinsicOp missing or incorrect intrinsic type attribute")
                .disp(ctx),
        )
    }
}

impl Parsable for CallIntrinsicOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::{optional, token};
        use pliron::irfmt::parsers::type_parser;

        let pos = state_stream.loc();

        let mut parser = (
            spaced(token('@').with(StringAttr::parser(()))),
            optional(spaced(FastmathFlagsAttr::parser(()))),
            delimited_list_parser('(', ')', ',', ssa_opd_parser()).skip(spaced(token(':'))),
            spaced(type_parser()),
        );

        let (iname, fmf, operands, ftype) = parser.parse_stream(state_stream).into_result()?.0;

        let ctx = &mut state_stream.state.ctx;
        let intr_ty = TypePtr::<FuncType>::from_ptr(ftype, ctx).map_err(|mut err| {
            err.set_loc(pos);
            err
        })?;
        let op = Self::new(ctx, iname, intr_ty, operands);
        if let Some(fmf) = fmf {
            op.set_attr_llvm_intrinsic_fastmath_flags(ctx, fmf);
        }
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

impl Verify for CallIntrinsicOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        // Check that the intrinsic name and type attributes are present.
        let Some(_name) = self.get_attr_llvm_intrinsic_name(ctx) else {
            return verify_err!(
                self.loc(ctx),
                CallIntrinsicVerifyErr::MissingIntrinsicNameAttr
            );
        };

        let Some(ty) = self.get_attr_llvm_intrinsic_type(ctx).and_then(|ty| {
            TypePtr::<FuncType>::from_ptr(TypedAttrInterface::get_type(&*ty, ctx), ctx).ok()
        }) else {
            return verify_err!(
                self.loc(ctx),
                CallIntrinsicVerifyErr::MissingIntrinsicTypeAttr
            );
        };

        let arg_types = ty.deref(ctx).arg_types();
        let res_type = ty.deref(ctx).result_type();

        // Check that the operand and result types match the intrinsic type.
        let op = &*self.op.deref(ctx);
        let intrinsic_arg_types = ty.deref(ctx).arg_types();
        if op.operands().count() != intrinsic_arg_types.len() {
            return verify_err!(self.loc(ctx), CallIntrinsicVerifyErr::OperandsMismatch);
        }

        for (i, operand) in op.operands().enumerate() {
            let opd_ty = operand.get_type(ctx);
            if opd_ty != arg_types[i] {
                return verify_err!(self.loc(ctx), CallIntrinsicVerifyErr::OperandsMismatch);
            }
        }

        let mut result_types = op.result_types();
        if let Some(result_type) = result_types.next()
            && result_type == res_type
            && result_types.next().is_none()
        {
        } else {
            return verify_err!(self.loc(ctx), CallIntrinsicVerifyErr::ResultsMismatch);
        }

        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all call operations.
pub fn register(ctx: &mut Context) {
    CallOp::register(ctx);
    CallIntrinsicOp::register(ctx);
}
