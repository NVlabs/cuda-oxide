/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Aggregate value operations - inserting and extracting from structs and arrays.
//!
//! This module contains LLVM dialect operations for aggregate value manipulation:
//!
//! ```text
//! ┌──────────────────┬────────────────┬────────────────────────────────────────┐
//! │ Operation        │ LLVM Opcode    │ Description                            │
//! ├──────────────────┼────────────────┼────────────────────────────────────────┤
//! │ InsertValueOp    │ insertvalue    │ Insert value into struct/array         │
//! │ ExtractValueOp   │ extractvalue   │ Extract value from struct/array        │
//! │ ExtractElementOp │ extractelement │ Extract element from vector (runtime)  │
//! └──────────────────┴────────────────┴────────────────────────────────────────┘
//! ```
//!
//! # InsertValue vs ExtractElement
//!
//! - `InsertValueOp` / `ExtractValueOp`: Use **compile-time constant** indices for structs/arrays
//! - `ExtractElementOp`: Uses a **runtime** index for vectors

use pliron::{
    arg_err_noloc,
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_op, derive_attr_get_set, derive_op_interface_impl},
    op::Op,
    operation::Operation,
    result::{Error, ErrorKind, Result},
    r#type::{TypeObj, Typed},
    value::Value,
    verify_err,
};

use crate::{
    attributes::InsertExtractValueIndicesAttr,
    types::{ArrayType, StructType},
};

// ============================================================================
// Error Types
// ============================================================================

/// Verification errors for insert/extract value operations.
#[derive(thiserror::Error, Debug)]
pub enum InsertExtractValueErr {
    #[error("Insert/Extract value instruction has no or incorrect indices attribute")]
    IndicesAttrErr,
    #[error("Invalid indices on insert/extract value instruction")]
    InvalidIndicesErr,
    #[error("Value being inserted / extracted does not match the type of the indexed aggregate")]
    ValueTypeErr,
}

/// Verification errors for [`ExtractElementOp`].
#[derive(thiserror::Error, Debug)]
pub enum ExtractElementErr {
    #[error("extractelement requires a vector or array type")]
    NotVectorType,
    #[error("extractelement result type does not match vector element type")]
    ResultTypeMismatch,
    #[error("extractelement index must be an integer type")]
    IndexNotInteger,
}

// ============================================================================
// InsertValue Operation
// ============================================================================

/// Insert a value into an aggregate at compile-time constant indices.
///
/// Creates a new aggregate with the specified element replaced.
///
/// Equivalent to LLVM's `insertvalue` instruction.
///
/// ### Operands
///
/// ```text
/// | operand     | description         |
/// |-------------|---------------------|
/// | `aggregate` | LLVM aggregate type |
/// | `value`     | LLVM type (to insert)|
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                        |
/// |--------|------------------------------------|
/// | `res`  | LLVM aggregate type (same as input)|
/// ```
#[def_op("llvm.insert_value")]
#[pliron::derive::format_op(
    "$0 attr($insert_value_indices, $InsertExtractValueIndicesAttr) `, ` $1 ` : ` type($0)"
)]
#[derive_attr_get_set(insert_value_indices : InsertExtractValueIndicesAttr)]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>)]
pub struct InsertValueOp;

impl InsertValueOp {
    /// Create a new [`InsertValueOp`].
    ///
    /// `aggregate` is the aggregate type and `value` is the value to insert.
    /// `indices` is the list of indices to insert the value at.
    pub fn new(ctx: &mut Context, aggregate: Value, value: Value, indices: Vec<u32>) -> Self {
        let result_type = aggregate.get_type(ctx);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![aggregate, value],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_insert_value_indices(ctx, InsertExtractValueIndicesAttr(indices));
        op
    }

    /// Get the indices for inserting value into aggregate.
    #[must_use]
    pub fn indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_attr_insert_value_indices(ctx).unwrap().clone().0
    }
}

impl Verify for InsertValueOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if self.get_attr_insert_value_indices(ctx).is_none() {
            verify_err!(loc.clone(), InsertExtractValueErr::IndicesAttrErr)?;
        }

        // Check that the value we are inserting is of the correct type.
        let aggr_type = self.get_operation().deref(ctx).get_operand(0).get_type(ctx);
        let indices = self.indices(ctx);
        match ExtractValueOp::indexed_type(ctx, aggr_type, &indices) {
            Err(e @ Error { .. }) => {
                return Err(Error {
                    kind: ErrorKind::VerificationFailed,
                    backtrace: std::backtrace::Backtrace::capture(),
                    ..e
                });
            }
            Ok(indexed_type) => {
                if indexed_type != self.get_operation().deref(ctx).get_operand(1).get_type(ctx) {
                    return verify_err!(loc, InsertExtractValueErr::ValueTypeErr);
                }
            }
        }

        Ok(())
    }
}

// ============================================================================
// ExtractValue Operation
// ============================================================================

/// Extract a value from an aggregate at compile-time constant indices.
///
/// Equivalent to LLVM's `extractvalue` instruction.
///
/// ### Operands
///
/// ```text
/// | operand     | description         |
/// |-------------|---------------------|
/// | `aggregate` | LLVM aggregate type |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description                    |
/// |--------|--------------------------------|
/// | `res`  | LLVM type (element at indices) |
/// ```
#[def_op("llvm.extract_value")]
#[pliron::derive::format_op(
    "$0 attr($extract_value_indices, $InsertExtractValueIndicesAttr) ` : ` type($0)"
)]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface)]
#[derive_attr_get_set(extract_value_indices : InsertExtractValueIndicesAttr)]
pub struct ExtractValueOp;

impl Verify for ExtractValueOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        if self.get_attr_extract_value_indices(ctx).is_none() {
            verify_err!(loc.clone(), InsertExtractValueErr::IndicesAttrErr)?;
        }
        // Check that the result type matches the indexed type
        let aggr_type = self.get_operation().deref(ctx).get_operand(0).get_type(ctx);
        let indices = self.indices(ctx);
        match Self::indexed_type(ctx, aggr_type, &indices) {
            Err(e @ Error { .. }) => {
                return Err(Error {
                    kind: ErrorKind::VerificationFailed,
                    backtrace: std::backtrace::Backtrace::capture(),
                    ..e
                });
            }
            Ok(indexed_type) => {
                if indexed_type != self.get_operation().deref(ctx).get_type(0) {
                    return verify_err!(loc, InsertExtractValueErr::ValueTypeErr);
                }
            }
        }

        Ok(())
    }
}

impl ExtractValueOp {
    /// Create a new [`ExtractValueOp`].
    ///
    /// `aggregate` is the aggregate type and `indices` is the list of indices.
    /// The result type is the type of the value at the given indices.
    pub fn new(ctx: &mut Context, aggregate: Value, indices: Vec<u32>) -> Result<Self> {
        let result_type = Self::indexed_type(ctx, aggregate.get_type(ctx), &indices)?;
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![aggregate],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_extract_value_indices(ctx, InsertExtractValueIndicesAttr(indices));
        Ok(op)
    }

    /// Get the indices for extracting value from aggregate.
    #[must_use]
    pub fn indices(&self, ctx: &Context) -> Vec<u32> {
        self.get_attr_extract_value_indices(ctx).unwrap().clone().0
    }

    /// Returns the type of the value at the given indices in the given aggregate type.
    pub fn indexed_type(
        ctx: &Context,
        aggr_type: Ptr<TypeObj>,
        indices: &[u32],
    ) -> Result<Ptr<TypeObj>> {
        fn indexed_type_inner(
            ctx: &Context,
            aggr_type: Ptr<TypeObj>,
            mut idx_itr: impl Iterator<Item = u32>,
        ) -> Result<Ptr<TypeObj>> {
            let Some(idx) = idx_itr.next() else {
                return Ok(aggr_type);
            };
            let aggr_type = &*aggr_type.deref(ctx);
            if let Some(st) = aggr_type.downcast_ref::<StructType>() {
                if st.is_opaque() || idx as usize >= st.num_fields() {
                    return arg_err_noloc!(InsertExtractValueErr::InvalidIndicesErr);
                }
                indexed_type_inner(ctx, st.field_type(idx as usize), idx_itr)
            } else if let Some(at) = aggr_type.downcast_ref::<ArrayType>() {
                if u64::from(idx) >= at.size() {
                    return arg_err_noloc!(InsertExtractValueErr::InvalidIndicesErr);
                }
                indexed_type_inner(ctx, at.elem_type(), idx_itr)
            } else {
                arg_err_noloc!(InsertExtractValueErr::InvalidIndicesErr)
            }
        }
        indexed_type_inner(ctx, aggr_type, indices.iter().copied())
    }
}

// ============================================================================
// ExtractElement Operation
// ============================================================================

/// Extract a scalar element from a vector at a runtime index.
///
/// Unlike `extractvalue`, this supports **runtime** indices.
///
/// Equivalent to LLVM's `extractelement` instruction.
///
/// ### Operands
///
/// ```text
/// | operand  | description                  |
/// |----------|------------------------------|
/// | `vector` | vector type                  |
/// | `index`  | integer type (runtime index) |
/// ```
///
/// ### Result(s):
///
/// ```text
/// | result | description         |
/// |--------|---------------------|
/// | `res`  | scalar element type |
/// ```
#[def_op("llvm.extractelement")]
#[pliron::derive::format_op("$0 ` [` $1 `] : ` type($0)")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>)]
pub struct ExtractElementOp;

impl ExtractElementOp {
    /// Create a new [`ExtractElementOp`].
    ///
    /// `vector` is the vector value and `index` is the runtime index.
    /// The result type is the element type of the vector.
    pub fn new(ctx: &mut Context, vector: Value, index: Value) -> Result<Self> {
        let vec_ty = vector.get_type(ctx);

        // Extract element type
        let result_type = {
            let vec_ty_ref = vec_ty.deref(ctx);
            if let Some(vt) = vec_ty_ref.downcast_ref::<crate::types::VectorType>() {
                vt.elem_type()
            } else if let Some(at) = vec_ty_ref.downcast_ref::<crate::types::ArrayType>() {
                // Also support arrays for flexibility
                at.elem_type()
            } else {
                return arg_err_noloc!(ExtractElementErr::NotVectorType);
            }
        };

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type],
            vec![vector, index],
            vec![],
            0,
        );
        Ok(Self { op })
    }
}

impl Verify for ExtractElementOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);

        let vec_val = op_ref.get_operand(0);
        let vec_ty = vec_val.get_type(ctx);
        let vec_ty_ref = vec_ty.deref(ctx);

        // Check that first operand is a vector or array type
        let elem_ty = if let Some(vt) = vec_ty_ref.downcast_ref::<crate::types::VectorType>() {
            vt.elem_type()
        } else if let Some(at) = vec_ty_ref.downcast_ref::<crate::types::ArrayType>() {
            at.elem_type()
        } else {
            return verify_err!(loc, ExtractElementErr::NotVectorType);
        };

        // Check that result type matches element type
        let result_ty = op_ref.get_type(0);
        if result_ty != elem_ty {
            return verify_err!(loc, ExtractElementErr::ResultTypeMismatch);
        }

        // Check that index is an integer type
        let idx_val = op_ref.get_operand(1);
        let idx_ty = idx_val.get_type(ctx);
        if idx_ty.deref(ctx).downcast_ref::<IntegerType>().is_none() {
            return verify_err!(loc, ExtractElementErr::IndexNotInteger);
        }

        Ok(())
    }
}

// ============================================================================
// Registration
// ============================================================================

/// Register all aggregate operations.
pub fn register(ctx: &mut Context) {
    InsertValueOp::register(ctx);
    ExtractValueOp::register(ctx);
    ExtractElementOp::register(ctx);
}
