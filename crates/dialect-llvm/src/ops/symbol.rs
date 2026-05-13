/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Symbol operations - functions, globals, and addressof.
//!
//! This module contains LLVM dialect operations for defining and referencing symbols:
//!
//! ```text
//! ┌─────────────┬───────────────────────────────────────────────┐
//! │ Operation   │ Description                                   │
//! ├─────────────┼───────────────────────────────────────────────┤
//! │ FuncOp      │ Function definition or declaration            │
//! │ GlobalOp    │ Global variable definition                    │
//! │ AddressOfOp │ Get pointer to a global or function           │
//! └─────────────┴───────────────────────────────────────────────┘
//! ```

use combine::parser::{Parser, char::spaces};
use pliron::{
    attribute::{AttrObj, AttributeDict},
    basic_block::BasicBlock,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::{IdentifierAttr, TypeAttr},
        op_interfaces::{
            self, ATTR_KEY_SYM_NAME, AtMostNRegionsInterface, AtMostOneRegionInterface,
            IsolatedFromAboveInterface, NOpdsInterface, NResultsInterface,
            SingleBlockRegionInterface, SymbolOpInterface, SymbolUserOpInterface,
        },
        type_interfaces::FunctionTypeInterface,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{op_interface_impl, pliron_op},
    identifier::Identifier,
    indented_block, input_err,
    irfmt::parsers::{attr_parser, spaced, type_parser},
    linked_list::ContainsLinkedList,
    location::{Located, Location},
    op::{Op, OpObj},
    operation::Operation,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream},
    printable::{self, Printable, indented_nl},
    region::Region,
    result::Result,
    symbol_table::SymbolTableCollection,
    r#type::{TypeObj, TypePtr},
    verify_err,
};

use crate::{
    attributes::{GlobalAddressSpaceAttr, LinkageAttr},
    op_interfaces::{IsDeclaration, LlvmSymbolName},
    types::{FuncType, PointerType},
};

use super::func_op_attr_names::ATTR_KEY_LLVM_FUNC_TYPE;
use super::global_op_attr_names::{ATTR_KEY_GLOBAL_INITIALIZER, ATTR_KEY_LLVM_GLOBAL_TYPE};

// ============================================================================
// Error Types
// ============================================================================

/// Verification errors for symbol user operations.
#[derive(thiserror::Error, Debug)]
pub enum SymbolUserOpVerifyErr {
    #[error("Symbol {0} not found")]
    SymbolNotFound(String),
    #[error("Function {0} should have been llvm.func type")]
    NotLlvmFunc(String),
    #[error("AddressOf Op can only refer to a function or a global variable")]
    AddressOfInvalidReference,
    #[error("Function call has incorrect type: {0}")]
    FuncTypeErr(String),
}

// ============================================================================
// Function Operation
// ============================================================================

/// Verification errors for [`FuncOp`].
#[derive(thiserror::Error, Debug)]
#[error("llvm.func op does not have llvm.func type")]
pub struct FuncOpTypeErr;

/// LLVM function definition or declaration.
///
/// Defines a function with its signature and optionally its body.
/// A function without a body is a declaration.
///
/// See upstream MLIR's [llvm.func](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmfunc-llvmllvmfuncop).
#[pliron_op(
    name = "llvm.func",
    interfaces = [
        SymbolOpInterface,
        IsolatedFromAboveInterface,
        AtMostNRegionsInterface<1>,
        AtMostOneRegionInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>,
        LlvmSymbolName
    ],
    attributes = (llvm_func_type: TypeAttr, llvm_function_linkage: LinkageAttr),
    verifier = "succ"
)]
pub struct FuncOp;

impl FuncOp {
    /// Create a new empty [`FuncOp`].
    pub fn new(ctx: &mut Context, name: Identifier, ty: TypePtr<FuncType>) -> Self {
        let ty_attr = TypeAttr::new(ty.into());
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let opop = FuncOp { op };
        opop.set_symbol_name(ctx, name);
        opop.set_attr_llvm_func_type(ctx, ty_attr);

        opop
    }

    /// Get the function signature (type).
    #[must_use]
    pub fn get_type(&self, ctx: &Context) -> TypePtr<FuncType> {
        let ty = TypedAttrInterface::get_type(&*self.get_attr_llvm_func_type(ctx).unwrap(), ctx);
        TypePtr::from_ptr(ty, ctx).unwrap()
    }

    /// Get the entry block (if it exists) of this function.
    #[must_use]
    pub fn get_entry_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        self.op
            .deref(ctx)
            .regions()
            .next()
            .and_then(|region| region.deref(ctx).get_head())
    }

    /// Get the entry block of this function, creating it if it does not exist.
    pub fn get_or_create_entry_block(&self, ctx: &mut Context) -> Ptr<BasicBlock> {
        if let Some(entry_block) = self.get_entry_block(ctx) {
            return entry_block;
        }

        // Create an empty entry block.
        assert!(
            self.op.deref(ctx).regions().next().is_none(),
            "FuncOp already has a region, but no block inside it"
        );
        let region = Operation::add_region(self.op, ctx);
        let arg_types = self.get_type(ctx).deref(ctx).arg_types().clone();
        let body = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), arg_types);
        body.insert_at_front(region, ctx);
        body
    }
}

impl pliron::r#type::Typed for FuncOp {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        self.get_type(ctx).into()
    }
}

impl Printable for FuncOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        use pliron::irfmt::printers::op::typed_symb_op_header;
        typed_symb_op_header(self).fmt(ctx, state, f)?;

        // Print attributes except for function type and symbol name.
        let mut attributes_to_print_separately =
            self.op.deref(ctx).attributes.clone_skip_outlined();
        attributes_to_print_separately
            .0
            .retain(|key, _| key != &*ATTR_KEY_LLVM_FUNC_TYPE && key != &*ATTR_KEY_SYM_NAME);
        indented_block!(state, {
            write!(
                f,
                "{}{}",
                indented_nl(state),
                attributes_to_print_separately.disp(ctx)
            )?;
        });

        if let Some(r) = self.get_region(ctx) {
            write!(f, " ")?;
            r.fmt(ctx, state, f)?;
        }
        Ok(())
    }
}

impl Parsable for FuncOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::{optional, token};

        if !results.is_empty() {
            input_err!(
                state_stream.loc(),
                op_interfaces::NResultsVerifyErr(0, results.len())
            )?;
        }

        let op = Operation::new(
            state_stream.state.ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![],
            vec![],
            0,
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
                let opop = FuncOp { op };
                opop.set_symbol_name(ctx, fname);
                opop.set_attr_llvm_func_type(ctx, ty_attr);
                OpObj::new(opop)
            })
            .into()
    }
}

impl IsDeclaration for FuncOp {
    fn is_declaration(&self, ctx: &Context) -> bool {
        self.get_region(ctx).is_none()
    }
}

// ============================================================================
// Global Operation
// ============================================================================

/// Verification errors for [`GlobalOp`].
#[derive(thiserror::Error, Debug)]
pub enum GlobalOpVerifyErr {
    #[error("GlobalOp must have a type")]
    MissingType,
}

/// Global variable definition.
///
/// Creates a global variable with the specified type.
/// An initializer can be specified either as an attribute or in a region.
///
/// Same as upstream MLIR's LLVM dialect [GlobalOp](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmlirglobal-llvmglobalop).
#[pliron_op(
    name = "llvm.global",
    interfaces = [
        IsolatedFromAboveInterface,
        NOpdsInterface<0>,
        NResultsInterface<0>,
        SymbolOpInterface,
        SingleBlockRegionInterface,
        LlvmSymbolName
    ],
    attributes = (
        llvm_global_type: TypeAttr,
        global_initializer,
        llvm_global_linkage: LinkageAttr,
        llvm_alignment: crate::attributes::AlignmentAttr,
        llvm_global_address_space: GlobalAddressSpaceAttr,
        // Raw initializer bytes (hex-encoded) propagated from mir-lower.
        // When present together with the relocations attr, the exporter
        // emits a packed-struct initializer combining the byte runs with
        // pointer relocations. See `crates/dialect-llvm/src/export.rs`
        // `export_global` for the formatting rules.
        llvm_initializer_bytes: pliron::builtin::attributes::StringAttr,
        // Cross-global pointer relocations: comma-separated `offset:name`
        // pairs (offset in bytes, name is the target global's LLVM name
        // without the leading `@`).
        llvm_initializer_relocations: pliron::builtin::attributes::StringAttr
    )
)]
pub struct GlobalOp;

impl GlobalOp {
    /// Create a new [`GlobalOp`]. An initializer region can be added later if needed.
    pub fn new(ctx: &mut Context, name: Identifier, ty: Ptr<TypeObj>) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let op = GlobalOp { op };
        op.set_symbol_name(ctx, name);
        op.set_attr_llvm_global_type(ctx, TypeAttr::new(ty));
        op.set_address_space(ctx, crate::types::address_space::GENERIC);
        op
    }

    /// Create a new [`GlobalOp`] in the specified address space.
    pub fn new_in_address_space(
        ctx: &mut Context,
        name: Identifier,
        ty: Ptr<TypeObj>,
        address_space: u32,
    ) -> Self {
        let op = Self::new(ctx, name, ty);
        op.set_address_space(ctx, address_space);
        op
    }

    /// Create a new [`GlobalOp`] with specified alignment.
    pub fn new_with_alignment(
        ctx: &mut Context,
        name: Identifier,
        ty: Ptr<TypeObj>,
        alignment: u64,
    ) -> Self {
        let op = Self::new(ctx, name, ty);
        op.set_alignment(ctx, alignment);
        op
    }

    /// Get alignment as u64 (returns None if not set).
    #[must_use]
    pub fn get_alignment(&self, ctx: &Context) -> Option<u64> {
        self.get_attr_llvm_alignment(ctx).map(|attr| attr.0 as u64)
    }

    /// Set alignment as u64.
    pub fn set_alignment(&self, ctx: &mut Context, alignment: u64) {
        self.set_attr_llvm_alignment(ctx, crate::attributes::AlignmentAttr(alignment as u32));
    }

    /// Get the global's address space.
    #[must_use]
    pub fn get_address_space(&self, ctx: &Context) -> u32 {
        self.get_attr_llvm_global_address_space(ctx)
            .map(|attr| attr.0)
            .unwrap_or(crate::types::address_space::GENERIC)
    }

    /// Set the global's address space.
    pub fn set_address_space(&self, ctx: &mut Context, address_space: u32) {
        self.set_attr_llvm_global_address_space(ctx, GlobalAddressSpaceAttr(address_space));
    }

    /// Get the initializer value of this global variable.
    #[must_use]
    pub fn get_initializer_value(&self, ctx: &Context) -> Option<AttrObj> {
        self.get_attr_global_initializer(ctx).map(|v| v.clone())
    }

    /// Get the initializer region's block of this global variable.
    #[must_use]
    pub fn get_initializer_block(&self, ctx: &Context) -> Option<Ptr<BasicBlock>> {
        (self.op.deref(ctx).num_regions() > 0).then(|| self.get_body(ctx, 0))
    }

    /// Get the initializer region of this global variable.
    #[must_use]
    pub fn get_initializer_region(&self, ctx: &Context) -> Option<Ptr<Region>> {
        (self.op.deref(ctx).num_regions() > 0)
            .then(|| self.get_operation().deref(ctx).get_region(0))
    }

    /// Set a simple initializer value for this global variable.
    pub fn set_initializer_value(&self, ctx: &Context, value: AttrObj) {
        self.set_attr_global_initializer(ctx, value);
    }

    /// Add an initializer region (with an entry block) for this global variable.
    pub fn add_initializer_region(&self, ctx: &mut Context) -> Ptr<Region> {
        assert!(
            self.get_initializer_value(ctx).is_none(),
            "Attempt to create an initializer region when there already is an initializer value"
        );
        let region = Operation::add_region(self.get_operation(), ctx);
        let entry = BasicBlock::new(ctx, Some("entry".try_into().unwrap()), vec![]);
        entry.insert_at_front(region, ctx);

        region
    }
}

impl pliron::r#type::Typed for GlobalOp {
    fn get_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        pliron::r#type::Typed::get_type(
            &*self
                .get_attr_llvm_global_type(ctx)
                .expect("GlobalOp missing or has incorrect type attribute"),
            ctx,
        )
    }
}

impl IsDeclaration for GlobalOp {
    fn is_declaration(&self, ctx: &Context) -> bool {
        self.get_initializer_value(ctx).is_none() && self.get_initializer_region(ctx).is_none()
    }
}

impl Verify for GlobalOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        if self.get_attr_llvm_global_type(ctx).is_none() {
            return verify_err!(loc, GlobalOpVerifyErr::MissingType);
        }

        // Check that there is at most one initializer
        if self.get_initializer_value(ctx).is_some() && self.get_initializer_region(ctx).is_some() {
            return verify_err!(loc, GlobalOpVerifyErr::MissingType);
        }

        Ok(())
    }
}

impl Printable for GlobalOp {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(
            f,
            "{} @{} : {}",
            self.get_opid(),
            self.get_symbol_name(ctx),
            <Self as pliron::r#type::Typed>::get_type(self, ctx).disp(ctx)
        )?;

        // Print attributes except for type, initializer and symbol name.
        let mut attributes_to_print_separately =
            self.op.deref(ctx).attributes.clone_skip_outlined();
        attributes_to_print_separately.0.retain(|key, _| {
            key != &*ATTR_KEY_LLVM_GLOBAL_TYPE
                && key != &*ATTR_KEY_SYM_NAME
                && key != &*ATTR_KEY_GLOBAL_INITIALIZER
        });
        indented_block!(state, {
            write!(
                f,
                "{}{}",
                indented_nl(state),
                attributes_to_print_separately.disp(ctx)
            )?;
        });

        if let Some(init_value) = self.get_initializer_value(ctx) {
            write!(f, " = {}", init_value.disp(ctx))?;
        }

        if let Some(init_region) = self.get_initializer_region(ctx) {
            write!(f, " = {}", init_region.print(ctx, state))?;
        }

        Ok(())
    }
}

impl Parsable for GlobalOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;
    fn parse<'a>(
        state_stream: &mut StateStream<'a>,
        results: Self::Arg,
    ) -> ParseResult<'a, Self::Parsed> {
        use combine::token;

        enum Initializer {
            Value(AttrObj),
            Region(Ptr<Region>),
        }

        let loc = state_stream.loc();
        if !results.is_empty() {
            return input_err!(loc, "GlobalOp must cannot have results")?;
        }
        let name_parser = token('@').with(Identifier::parser(()));
        let type_parser = type_parser();
        let attr_dict_parser = AttributeDict::parser(());

        let mut parser = name_parser
            .skip(spaced(token(':')))
            .and(type_parser)
            .and(spaced(attr_dict_parser));

        let (((name, ty), attr_dict), _) = parser.parse_stream(state_stream).into_result()?;
        let op = GlobalOp::new(state_stream.state.ctx, name, ty);
        op.get_operation()
            .deref_mut(state_stream.state.ctx)
            .attributes
            .0
            .extend(attr_dict.0);

        // Parse optional initializer value or region.
        let initializer_parser = token('=').skip(spaces()).with(
            attr_parser()
                .map(Initializer::Value)
                .or(Region::parser(op.get_operation()).map(Initializer::Region)),
        );

        let initializer = spaces()
            .with(combine::optional(initializer_parser))
            .parse_stream(state_stream)
            .into_result()?;

        if let Some(initializer) = initializer.0 {
            match initializer {
                Initializer::Value(v) => op.set_initializer_value(state_stream.state.ctx, v),
                Initializer::Region(_r) => {
                    // Nothing to do since the region is already added to the operation during parsing.
                }
            }
        }

        Ok(OpObj::new(op)).into_parse_result()
    }
}

// ============================================================================
// AddressOf Operation
// ============================================================================

/// Get a pointer to a global variable or function.
///
/// Creates an SSA value containing a pointer to a global value (function, variable, etc).
///
/// Same as upstream MLIR's LLVM dialect [AddressOfOp](https://mlir.llvm.org/docs/Dialects/LLVM/#llvmmliraddressof-llvmaddressofop).
///
/// ### Results:
///
/// ```text
/// | result   | description       |
/// |----------|-------------------|
/// | `result` | LLVM pointer type |
/// ```
#[pliron_op(
    name = "llvm.addressof",
    format = "`@` attr($global_name, $IdentifierAttr) ` : ` type($0)",
    interfaces = [NResultsInterface<1>, pliron::builtin::op_interfaces::OneResultInterface, NOpdsInterface<0>],
    attributes = (global_name: IdentifierAttr)
)]
pub struct AddressOfOp;

impl Verify for AddressOfOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::builtin::op_interfaces::OneResultInterface;

        let loc = self.loc(ctx);

        // Check result is pointer
        let res_ty = self.result_type(ctx);
        if !res_ty.deref(ctx).is::<PointerType>() {
            return verify_err!(loc, "AddressOfOp result must be a pointer");
        }

        // Check symbol name exists
        if self.get_attr_global_name(ctx).is_none() {
            return verify_err!(loc, "AddressOfOp must have a global_name attribute");
        }
        Ok(())
    }
}

impl AddressOfOp {
    /// Create a new [`AddressOfOp`] with explicit address space.
    ///
    /// The address space should match the global variable's memory space.
    pub fn new(ctx: &mut Context, global_name: Identifier, address_space: u32) -> Self {
        let result_type = PointerType::get(ctx, address_space).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![],
            vec![],
            0,
        );
        let op = AddressOfOp { op };
        op.set_attr_global_name(ctx, IdentifierAttr::new(global_name));
        op
    }

    /// Get the global name that this refers to.
    #[must_use]
    pub fn get_global_name(&self, ctx: &Context) -> Identifier {
        self.get_attr_global_name(ctx)
            .expect("AddressOfOp missing or has incorrect global_name attribute type")
            .clone()
            .into()
    }

    /// If this operation refers to a global, get it.
    pub fn get_global(
        &self,
        ctx: &Context,
        symbol_tables: &mut SymbolTableCollection,
    ) -> Option<GlobalOp> {
        let global_name = self.get_global_name(ctx);
        symbol_tables
            .lookup_symbol_in_nearest_table(ctx, self.get_operation(), &global_name)
            .and_then(|sym_op| {
                (sym_op as Box<dyn Op>)
                    .downcast::<GlobalOp>()
                    .map(|op| *op)
                    .ok()
            })
    }

    /// If this operation refers to a function, get it.
    pub fn get_function(
        &self,
        ctx: &Context,
        symbol_tables: &mut SymbolTableCollection,
    ) -> Option<FuncOp> {
        let global_name = self.get_global_name(ctx);
        symbol_tables
            .lookup_symbol_in_nearest_table(ctx, self.get_operation(), &global_name)
            .and_then(|sym_op| {
                (sym_op as Box<dyn Op>)
                    .downcast::<FuncOp>()
                    .map(|op| *op)
                    .ok()
            })
    }
}

#[op_interface_impl]
impl SymbolUserOpInterface for AddressOfOp {
    fn used_symbols(&self, ctx: &Context) -> Vec<Identifier> {
        vec![self.get_global_name(ctx)]
    }

    fn verify_symbol_uses(
        &self,
        ctx: &Context,
        symbol_tables: &mut SymbolTableCollection,
    ) -> Result<()> {
        let loc = self.loc(ctx);
        let global_name = self.get_global_name(ctx);
        let Some(symbol) =
            symbol_tables.lookup_symbol_in_nearest_table(ctx, self.get_operation(), &global_name)
        else {
            return verify_err!(
                loc,
                SymbolUserOpVerifyErr::SymbolNotFound(global_name.to_string())
            );
        };

        // Symbol can only be a FuncOp or a GlobalOp
        let is_global = (&*symbol as &dyn Op).is::<GlobalOp>();
        let is_func = (&*symbol as &dyn Op).is::<FuncOp>();
        if !is_global && !is_func {
            return verify_err!(loc, SymbolUserOpVerifyErr::AddressOfInvalidReference);
        }

        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all symbol operations.
pub fn register(ctx: &mut Context) {
    FuncOp::register(ctx);
    GlobalOp::register(ctx);
    AddressOfOp::register(ctx);
}
