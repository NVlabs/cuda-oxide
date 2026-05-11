/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Memory operations - allocation, load, store, and pointer arithmetic.
//!
//! This module contains LLVM dialect operations for memory access:
//!
//! ```text
//! ┌─────────────────┬───────────────┬─────────────────────────────────────────┐
//! │ Operation       │ LLVM Opcode   │ Description                             │
//! ├─────────────────┼───────────────┼─────────────────────────────────────────┤
//! │ AllocaOp        │ alloca        │ Stack allocation                        │
//! │ LoadOp          │ load          │ Load value from memory                  │
//! │ StoreOp         │ store         │ Store value to memory                   │
//! │ GetElementPtrOp │ getelementptr │ Pointer arithmetic / struct field access│
//! └─────────────────┴───────────────┴─────────────────────────────────────────┘
//! ```

use pliron::{
    arg_err_noloc,
    builtin::{
        attr_interfaces::TypedAttrInterface,
        attributes::TypeAttr,
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Result,
    r#type::{TypeObj, Typed},
    utils::vec_exns::VecExtns,
    value::Value,
    verify_err,
};

use crate::{
    attributes::{GepIndexAttr, GepIndicesAttr},
    op_interfaces::PointerTypeResult,
    types::{ArrayType, PointerType, StructType},
};

// ============================================================================
// Stack Allocation
// ============================================================================

/// Verification errors for [`AllocaOp`].
#[derive(thiserror::Error, Debug)]
pub enum AllocaOpVerifyErr {
    #[error("Operand must be a signless integer")]
    OperandType,
    #[error("Missing or incorrect type of attribute for element type")]
    ElemTypeAttr,
}

/// Stack allocation operation.
///
/// Allocates memory on the stack for the given element type and count.
/// Returns a pointer to the allocated memory in generic address space.
///
/// Equivalent to LLVM's `alloca` instruction.
///
/// ### Operands
///
/// ```text
/// | operand      | description                          |
/// |--------------|--------------------------------------|
/// | `array_size` | Signless integer (number of elements)|
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description   |
/// |--------|---------------|
/// | `res`  | [PointerType] |
/// ```
#[pliron_op(
    name = "llvm.alloca",
    format = "`[` attr($alloca_element_type, $TypeAttr) ` x ` $0 `]` ` : ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface],
    attributes = (alloca_element_type: TypeAttr)
)]
pub struct AllocaOp;

impl Verify for AllocaOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if !self.operand_type(ctx).deref(ctx).is::<IntegerType>() {
            return verify_err!(loc, AllocaOpVerifyErr::OperandType);
        }
        if self.get_attr_alloca_element_type(ctx).is_none() {
            verify_err!(loc, AllocaOpVerifyErr::ElemTypeAttr)?;
        }
        Ok(())
    }
}

#[pliron::derive::op_interface_impl]
impl PointerTypeResult for AllocaOp {
    fn result_pointee_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        TypedAttrInterface::get_type(
            &*self
                .get_attr_alloca_element_type(ctx)
                .expect("AllocaOp missing or incorrect type for elem_type attribute"),
            ctx,
        )
    }
}

impl AllocaOp {
    /// Create a new [`AllocaOp`] that allocates on the stack.
    ///
    /// Returns a pointer in generic address space (alloca returns local memory).
    pub fn new(ctx: &mut Context, elem_type: Ptr<TypeObj>, size: Value) -> Self {
        let ptr_ty = PointerType::get_generic(ctx).into();
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![ptr_ty],
            vec![size],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_alloca_element_type(ctx, TypeAttr::new(elem_type));
        op
    }
}

// ============================================================================
// Load Operation
// ============================================================================

/// Verification errors for [`LoadOp`].
#[derive(thiserror::Error, Debug)]
pub enum LoadOpVerifyErr {
    #[error("Load operand must be a pointer")]
    OperandTypeErr,
}

/// Load value from memory.
///
/// Reads a value from the memory location pointed to by the operand.
///
/// Equivalent to LLVM's `load` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description     |
/// |---------|-----------------|
/// | `addr`  | [`PointerType`] |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description     |
/// |--------|-----------------|
/// | `res`  | sized LLVM type |
/// ```
#[pliron_op(
    name = "llvm.load",
    format = "$0 ` : ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface]
)]
pub struct LoadOp;

impl LoadOp {
    /// Create a new [`LoadOp`].
    pub fn new(ctx: &mut Context, ptr: Value, res_ty: Ptr<TypeObj>) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![res_ty],
                vec![ptr],
                vec![],
                0,
            ),
        }
    }
}

impl Verify for LoadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if !self.operand_type(ctx).deref(ctx).is::<PointerType>() {
            return verify_err!(loc, LoadOpVerifyErr::OperandTypeErr);
        }
        Ok(())
    }
}

// ============================================================================
// Store Operation
// ============================================================================

/// Verification errors for [`StoreOp`].
#[derive(thiserror::Error, Debug)]
pub enum StoreOpVerifyErr {
    #[error("Store operand must have two operands")]
    NumOpdsErr,
    #[error("Store operand must have a pointer as its second argument")]
    AddrOpdTypeErr,
}

/// Store value to memory.
///
/// Writes a value to the memory location pointed to by the address operand.
///
/// Equivalent to LLVM's `store` instruction.
///
/// ### Operands
///
/// ```text
/// | operand | description                 |
/// |---------|-----------------------------|
/// | `value` | Sized type (value to store) |
/// | `addr`  | [PointerType] (destination) |
/// ```
#[pliron_op(
    name = "llvm.store",
    format = "`*` $1 ` <- ` $0",
    interfaces = [NResultsInterface<0>, NOpdsInterface<2>]
)]
pub struct StoreOp;

impl StoreOp {
    /// Create a new [`StoreOp`].
    pub fn new(ctx: &mut Context, value: Value, ptr: Value) -> Self {
        Self {
            op: Operation::new(
                ctx,
                Self::get_concrete_op_info(),
                vec![],
                vec![value, ptr],
                vec![],
                0,
            ),
        }
    }

    /// Get the value operand.
    #[must_use]
    pub fn value_opd(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_operand(0)
    }

    /// Get the address operand.
    #[must_use]
    pub fn address_opd(&self, ctx: &Context) -> Value {
        self.op.deref(ctx).get_operand(1)
    }
}

impl Verify for StoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op = &*self.op.deref(ctx);

        if !op
            .get_operand(1)
            .get_type(ctx)
            .deref(ctx)
            .is::<PointerType>()
        {
            return verify_err!(loc, StoreOpVerifyErr::AddrOpdTypeErr);
        }
        Ok(())
    }
}

// ============================================================================
// GetElementPtr Operation
// ============================================================================

/// A GEP index can be either a compile-time constant or a runtime SSA value.
#[derive(Clone)]
pub enum GepIndex {
    /// Compile-time constant index.
    Constant(u32),
    /// Runtime SSA value index.
    Value(Value),
}

impl Printable for GepIndex {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Self::Constant(c) => write!(f, "{c}"),
            Self::Value(v) => write!(f, "{}", v.disp(ctx)),
        }
    }
}

/// Verification errors for [`GetElementPtrOp`].
#[derive(thiserror::Error, Debug)]
pub enum GetElementPtrOpErr {
    #[error("GetElementPtrOp has no or incorrect indices attribute")]
    IndicesAttrErr,
    #[error("The indices on this GEP are invalid for its source element type")]
    IndicesErr,
}

/// Pointer arithmetic and struct field access.
///
/// Computes a pointer to an element within an aggregate (struct, array) starting
/// from a base pointer. The indices specify the path through nested aggregates.
///
/// Equivalent to LLVM's `getelementptr` instruction.
///
/// ### Operands
///
/// ```text
/// | operand          | description                      |
/// |------------------|----------------------------------|
/// | `base`           | LLVM pointer type                |
/// | `dynamicIndices` | Any number of signless integers  |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description       |
/// |--------|-------------------|
/// | `res`  | LLVM pointer type |
/// ```
#[pliron_op(
    name = "llvm.gep",
    format = "`<` attr($gep_src_elem_type, $TypeAttr) `>` ` (` operands(CharSpace(`,`)) `)` attr($gep_indices, $GepIndicesAttr) ` : ` type($0)",
    interfaces = [NResultsInterface<1>, OneResultInterface],
    attributes = (gep_src_elem_type: TypeAttr, gep_indices: GepIndicesAttr)
)]
pub struct GetElementPtrOp;

#[pliron::derive::op_interface_impl]
impl PointerTypeResult for GetElementPtrOp {
    fn result_pointee_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        Self::indexed_type(ctx, self.src_elem_type(ctx), &self.indices(ctx))
            .expect("Invalid indices for GEP")
    }
}

impl Verify for GetElementPtrOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use pliron::result::{Error, ErrorKind};

        let loc = self.loc(ctx);
        if self.get_attr_gep_indices(ctx).is_none() {
            verify_err!(loc, GetElementPtrOpErr::IndicesAttrErr)?;
        }

        if let Err(e @ Error { .. }) =
            Self::indexed_type(ctx, self.src_elem_type(ctx), &self.indices(ctx))
        {
            return Err(Error {
                kind: ErrorKind::VerificationFailed,
                backtrace: std::backtrace::Backtrace::capture(),
                ..e
            });
        }

        Ok(())
    }
}

impl GetElementPtrOp {
    /// Create a new [`GetElementPtrOp`].
    ///
    /// The result pointer preserves the address space of the base pointer.
    pub fn new(
        ctx: &mut Context,
        base: Value,
        indices: Vec<GepIndex>,
        src_elem_type: Ptr<TypeObj>,
    ) -> Result<Self> {
        // Preserve address space from base pointer
        let base_ty = base.get_type(ctx);
        let address_space = if let Some(ptr_ty) = base_ty.deref(ctx).downcast_ref::<PointerType>() {
            ptr_ty.address_space
        } else {
            0 // Default to generic if not a pointer type
        };
        let result_type = PointerType::get(ctx, address_space).into();
        let mut attr: Vec<GepIndexAttr> = Vec::new();
        let mut opds: Vec<Value> = vec![base];
        for idx in indices {
            match idx {
                GepIndex::Constant(c) => {
                    attr.push(GepIndexAttr::Constant(c));
                }
                GepIndex::Value(v) => {
                    attr.push(GepIndexAttr::OperandIdx(opds.push_back(v)));
                }
            }
        }
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            opds,
            vec![],
            0,
        );
        let src_elem_type = TypeAttr::new(src_elem_type);
        let op = Self { op };

        op.set_attr_gep_indices(ctx, GepIndicesAttr(attr));
        op.set_attr_gep_src_elem_type(ctx, src_elem_type);
        Ok(op)
    }

    /// Get the source pointer's element type.
    #[must_use]
    pub fn src_elem_type(&self, ctx: &Context) -> Ptr<TypeObj> {
        TypedAttrInterface::get_type(
            &*self
                .get_attr_gep_src_elem_type(ctx)
                .expect("GetElementPtrOp missing or has incorrect src_elem_type attribute type"),
            ctx,
        )
    }

    /// Get the base (source) pointer of this GEP.
    #[must_use]
    pub fn src_ptr(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the indices of this GEP.
    #[must_use]
    pub fn indices(&self, ctx: &Context) -> Vec<GepIndex> {
        let op = &*self.op.deref(ctx);
        self.get_attr_gep_indices(ctx)
            .unwrap()
            .0
            .iter()
            .map(|index| match index {
                GepIndexAttr::Constant(c) => GepIndex::Constant(*c),
                GepIndexAttr::OperandIdx(i) => GepIndex::Value(op.get_operand(*i)),
            })
            .collect()
    }

    /// Returns the result element type of a GEP with the given source element type and indexes.
    ///
    /// See [getIndexedType](https://llvm.org/doxygen/classllvm_1_1GetElementPtrInst.html#a99d4bfe49182f8d80abb1960f2c12d46)
    pub fn indexed_type(
        ctx: &Context,
        src_elem_type: Ptr<TypeObj>,
        indices: &[GepIndex],
    ) -> Result<Ptr<TypeObj>> {
        fn indexed_type_inner(
            ctx: &Context,
            src_elem_type: Ptr<TypeObj>,
            mut idx_itr: impl Iterator<Item = GepIndex>,
        ) -> Result<Ptr<TypeObj>> {
            let Some(idx) = idx_itr.next() else {
                return Ok(src_elem_type);
            };
            let src_elem_type = &*src_elem_type.deref(ctx);
            if let Some(st) = src_elem_type.downcast_ref::<StructType>() {
                let GepIndex::Constant(i) = idx else {
                    return arg_err_noloc!(GetElementPtrOpErr::IndicesErr);
                };
                if st.is_opaque() || i as usize >= st.num_fields() {
                    return arg_err_noloc!(GetElementPtrOpErr::IndicesErr);
                }
                indexed_type_inner(ctx, st.field_type(i as usize), idx_itr)
            } else if let Some(at) = src_elem_type.downcast_ref::<ArrayType>() {
                indexed_type_inner(ctx, at.elem_type(), idx_itr)
            } else {
                arg_err_noloc!(GetElementPtrOpErr::IndicesErr)
            }
        }
        // The first index is for the base (source) pointer. Skip that.
        indexed_type_inner(ctx, src_elem_type, indices.iter().skip(1).cloned())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all memory operations.
pub fn register(ctx: &mut Context) {
    AllocaOp::register(ctx);
    LoadOp::register(ctx);
    StoreOp::register(ctx);
    GetElementPtrOp::register(ctx);
}
