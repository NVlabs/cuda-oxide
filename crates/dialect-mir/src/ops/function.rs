/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR function operations.
//!
//! This module defines the function operation for the MIR dialect.

use combine::{Parser, optional, token};
use once_cell::sync::Lazy;
use pliron::{
    attribute::AttributeDict,
    attribute::attr_cast,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::TypeAttr,
        op_interfaces::{
            ATTR_KEY_SYM_NAME, IsolatedFromAboveInterface, NOpdsInterface, NRegionsInterface,
            NResultsInterface, OneRegionInterface, SymbolOpInterface,
        },
        type_interfaces::FunctionTypeInterface,
        types::FunctionType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    identifier::Identifier,
    indented_block, input_err,
    irfmt::{
        parsers::{spaced, type_parser},
        printers::op::{region, typed_symb_op_header},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{Parsable, ParseResult, StateStream},
    printable::{Printable, State, indented_nl},
    region::Region,
    result::Error,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast},
    verify_err,
};
use pliron_derive::pliron_op;

/// MIR function operation.
///
/// Represents a function in MIR. Contains a single region with basic blocks.
///
/// # Attributes
///
/// ```text
/// | Name           | Type      | Description                        |
/// |----------------|-----------|------------------------------------|
/// | `sym_name`     | StringAttr| Function name (from SymbolOpInterface) |
/// | `mir_func_type`| TypeAttr  | Function type (mir.func_type)      |
/// ```
///
/// # Verification
///
/// - Must have a `mir_func_type` attribute that implements `FunctionTypeInterface`.
/// - The entry block arguments must match the function input types.
#[pliron_op(
    name = "mir.func",
    interfaces = [
        SymbolOpInterface,
        IsolatedFromAboveInterface,
        NRegionsInterface<1>,
        OneRegionInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>
    ],
    attributes = (mir_func_type: TypeAttr)
)]
pub struct MirFuncOp;

impl MirFuncOp {
    /// Create a new MirFuncOp.
    pub fn new(ctx: &mut Context, op_ptr: Ptr<Operation>, func_type_attr: TypeAttr) -> Self {
        let op = MirFuncOp { op: op_ptr };
        op.set_attr_mir_func_type(ctx, func_type_attr);
        op
    }

    /// Create a MirFuncOp from an existing operation pointer.
    ///
    /// Returns `None` if the operation is not a `mir.func`.
    pub fn wrap(ctx: &Context, op: Ptr<Operation>) -> Option<Self> {
        if Operation::get_opid(op, ctx) == Self::get_opid_static() {
            Some(MirFuncOp { op })
        } else {
            None
        }
    }

    /// Get the function type.
    pub fn get_type(&self, ctx: &Context) -> TypedHandle<FunctionType> {
        let ty = attr_cast::<dyn TypedAttrInterface>(&*self.get_attr_mir_func_type(ctx).unwrap())
            .unwrap()
            .get_type(ctx);
        TypedHandle::from_handle(ty, ctx).unwrap()
    }
}

impl Typed for MirFuncOp {
    fn get_type(&self, ctx: &Context) -> TypeHandle {
        self.get_type(ctx).into()
    }
}

impl Printable for MirFuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        typed_symb_op_header(self).fmt(ctx, state, f)?;
        let mut attributes_to_print_separately = self
            .get_operation()
            .deref(ctx)
            .attributes
            .clone_skip_outlined();
        attributes_to_print_separately
            .0
            .retain(|key, _| key != &*ATTR_KEY_MIR_FUNC_TYPE && key != &*ATTR_KEY_SYM_NAME);

        if !attributes_to_print_separately.0.is_empty() {
            indented_block!(state, {
                write!(f, "{}", indented_nl(state))?;
                attributes_to_print_separately.fmt(ctx, state, f)?;
            });
        }
        write!(f, " ")?;
        region(self).fmt(ctx, state, f)?;
        Ok(())
    }
}

impl Parsable for MirFuncOp {
    type Arg = Vec<(Identifier, pliron::location::Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                pliron::builtin::op_interfaces::NResultsVerifyErr(0, results.len())
            )?
        }
        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            1,
        );
        let mut parser = (
            spaced(token('@').with(Identifier::parser(()))).skip(spaced(token(':'))),
            spaced(type_parser()),
            spaced(AttributeDict::parser(())),
            spaced(optional(Region::parser(op))),
        );
        parser
            .parse_stream(state_stream)
            .map(|(fname, fty, attrs, _region)| -> OpObj {
                let ctx = &mut state_stream.state.ctx;
                op.deref_mut(ctx).attributes = attrs;
                let ty_attr = TypeAttr::new(fty);
                let opop = MirFuncOp { op };
                opop.set_symbol_name(ctx, fname);
                opop.set_attr_mir_func_type(ctx, ty_attr);
                OpObj::new(opop)
            })
            .into()
    }
}

impl Verify for MirFuncOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Verify function type attribute
        let func_ty = self.get_type(ctx);
        let func_ty_ref = func_ty.deref(ctx);

        // Check inputs via interface
        let interface = match type_cast::<dyn FunctionTypeInterface>(&*func_ty_ref) {
            Some(i) => i,
            None => {
                return verify_err!(
                    op.loc(),
                    "FunctionType does not implement FunctionTypeInterface"
                );
            }
        };

        // Verify region arguments match function type inputs
        let region = op.get_region(0).deref(ctx);

        // Check if there is an entry block
        if let Some(entry_block_ptr) = region.get_head() {
            let entry_block = entry_block_ptr.deref(ctx);
            let inputs = interface.arg_types();

            if entry_block.get_num_arguments() != inputs.len() {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp entry block argument count must match function type inputs"
                );
            }

            for (i, arg) in entry_block.arguments().enumerate() {
                if arg.get_type(ctx) != inputs[i] {
                    return verify_err!(
                        op.loc(),
                        "MirFuncOp entry block argument {} type mismatch",
                        i
                    );
                }
            }
        }

        Ok(())
    }
}

/// Attribute key for the MIR function type.
pub static ATTR_KEY_MIR_FUNC_TYPE: Lazy<Identifier> =
    Lazy::new(|| "mir_func_type".try_into().unwrap());

/// Register function operations into the given context.
pub fn register(ctx: &mut Context) {
    MirFuncOp::register(ctx);
}
