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
        attributes::{StringAttr, TypeAttr},
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

use crate::types::{MirPointerKind, MirPtrType, MirSliceType};

/// Per-source-argument marker populated only after rustc has proved the LLVM
/// `noalias` contract for that Rust reference. The index is the MIR function
/// argument index before aggregate flattening.
pub const MIR_PARAM_NOALIAS_ATTR_PREFIX: &str = "mir_param_noalias_";

/// Per-source-argument marker populated only after rustc has proved the LLVM
/// `readonly` contract for that Rust shared reference. The index is the MIR
/// function argument index before aggregate flattening.
pub const MIR_PARAM_READONLY_ATTR_PREFIX: &str = "mir_param_readonly_";

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
/// | `mir_param_noalias_N` | StringAttr | rustc-proven `noalias` for source arg N |
/// | `mir_param_readonly_N` | StringAttr | rustc-proven `readonly` for source arg N |
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

    /// Record rustc-proven aliasing facts for one source-level function argument.
    ///
    /// These facts are attached to the MIR argument index, before slices are
    /// flattened into `(ptr, len)` by `mir-lower`. `readonly` is intentionally
    /// represented separately from pointer kind: `SharedRef` alone is not
    /// sufficient when the pointee contains `UnsafeCell`.
    pub fn set_reference_param_attrs(
        &self,
        ctx: &mut Context,
        index: usize,
        noalias: bool,
        readonly: bool,
    ) {
        if noalias {
            set_param_marker(
                ctx,
                self.get_operation(),
                MIR_PARAM_NOALIAS_ATTR_PREFIX,
                index,
            );
        }
        if readonly {
            set_param_marker(
                ctx,
                self.get_operation(),
                MIR_PARAM_READONLY_ATTR_PREFIX,
                index,
            );
        }
    }

    /// Whether rustc proved that this source-level argument may carry LLVM `noalias`.
    pub fn param_noalias(&self, ctx: &Context, index: usize) -> bool {
        has_param_marker(
            ctx,
            self.get_operation(),
            MIR_PARAM_NOALIAS_ATTR_PREFIX,
            index,
        )
    }

    /// Whether rustc proved that this source-level argument may carry LLVM `readonly`.
    pub fn param_readonly(&self, ctx: &Context, index: usize) -> bool {
        has_param_marker(
            ctx,
            self.get_operation(),
            MIR_PARAM_READONLY_ATTR_PREFIX,
            index,
        )
    }
}

fn param_marker_key(prefix: &str, index: usize) -> Identifier {
    format!("{prefix}{index}")
        .as_str()
        .try_into()
        .expect("parameter marker attribute name is valid")
}

fn set_param_marker(ctx: &mut Context, op: Ptr<Operation>, prefix: &str, index: usize) {
    let key = param_marker_key(prefix, index);
    op.deref_mut(ctx)
        .attributes
        .set(key, StringAttr::new("true".to_string()));
}

fn has_param_marker(ctx: &Context, op: Ptr<Operation>, prefix: &str, index: usize) -> bool {
    let key = param_marker_key(prefix, index);
    op.deref(ctx).attributes.get::<StringAttr>(&key).is_some()
}

fn marker_index(key: &Identifier, prefix: &str) -> Option<Result<usize, ()>> {
    let key = key.to_string();
    key.strip_prefix(prefix)
        .map(|suffix| suffix.parse::<usize>().map_err(|_| ()))
}

fn pointer_kind_for_function_input(ctx: &Context, ty: TypeHandle) -> Option<MirPointerKind> {
    let ty = ty.deref(ctx);
    if let Some(pointer) = ty.downcast_ref::<MirPtrType>() {
        Some(pointer.pointer_kind())
    } else {
        ty.downcast_ref::<MirSliceType>()
            .map(MirSliceType::pointer_kind)
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

        let inputs = interface.arg_types();

        // Reference-derived optimizer facts are deliberately audited here,
        // before MIR-to-LLVM flattening. Raw/erased pointers must never acquire
        // Rust reference guarantees, and `readonly` is only valid for a shared
        // reference for which the importer also proved `noalias`.
        for key in op.attributes.0.keys() {
            let marker = marker_index(key, MIR_PARAM_NOALIAS_ATTR_PREFIX)
                .map(|index| (index, false))
                .or_else(|| {
                    marker_index(key, MIR_PARAM_READONLY_ATTR_PREFIX).map(|index| (index, true))
                });
            let Some((index, is_readonly)) = marker else {
                continue;
            };
            let index = match index {
                Ok(index) => index,
                Err(()) => {
                    return verify_err!(
                        op.loc(),
                        "MirFuncOp has malformed reference-parameter attribute `{}`",
                        key
                    );
                }
            };
            if index >= inputs.len() {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp reference-parameter attribute `{}` indexes argument {}, but the function has only {} arguments",
                    key,
                    index,
                    inputs.len()
                );
            }

            let kind = pointer_kind_for_function_input(ctx, inputs[index]);
            if is_readonly {
                if kind != Some(MirPointerKind::SharedRef) {
                    return verify_err!(
                        op.loc(),
                        "MirFuncOp readonly proof on argument {} requires a SharedRef pointer kind",
                        index
                    );
                }
                if !self.param_noalias(ctx, index) {
                    return verify_err!(
                        op.loc(),
                        "MirFuncOp readonly proof on argument {} must carry the matching noalias proof",
                        index
                    );
                }
            } else if !matches!(
                kind,
                Some(MirPointerKind::SharedRef | MirPointerKind::UniqueRef)
            ) {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp noalias proof on argument {} requires a Rust reference pointer kind",
                    index
                );
            } else if kind == Some(MirPointerKind::SharedRef) && !self.param_readonly(ctx, index) {
                return verify_err!(
                    op.loc(),
                    "MirFuncOp SharedRef noalias proof on argument {} must carry the matching readonly proof",
                    index
                );
            }
        }

        // Verify region arguments match function type inputs
        let region = op.get_region(0).deref(ctx);

        // Check if there is an entry block
        if let Some(entry_block_ptr) = region.get_head() {
            let entry_block = entry_block_ptr.deref(ctx);
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
