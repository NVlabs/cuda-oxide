/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! MIR aggregate operations.
//!
//! This module defines struct and tuple manipulation operations for the MIR dialect.

use pliron::{
    builtin::{
        op_interfaces::{NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface},
        types::IntegerType,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    printable::Printable,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

use crate::attributes::FieldIndexAttr;
use crate::types::{
    MirArrayType, MirDisjointSliceType, MirPtrType, MirSliceType, MirStructType, MirTupleType,
    MirUnionType,
};

// ============================================================================
// MirExtractFieldOp
// ============================================================================

/// MIR extract field/element operation.
///
/// Extracts a field from a tuple, slice, disjoint slice, struct, array (constant index),
/// or scalar-lowered newtype.
///
/// # Attributes
///
/// ```text
/// | Name    | Type           | Description                |
/// |---------|----------------|----------------------------|
/// | `index` | FieldIndexAttr | Index of field to extract  |
/// ```
///
/// # Operands
///
/// ```text
/// | Name      | Type                                                    |
/// |-----------|---------------------------------------------------------|
/// | `operand` | MirTupleType, MirSliceType, MirStructType, MirArrayType |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                          |
/// |-------|-------------------------------|
/// | `res` | Type of extracted field/elem  |
/// ```
///
/// # Verification
///
/// - Operand must be tuple, slice, disjoint slice, struct, array, or scalar (newtype).
/// - Index must be valid (compile-time constant) for the type.
/// - Result type must match the extracted field/element type.
///
/// # Note on Arrays
///
/// For arrays with constant indices, this op lowers to `llvm.extractvalue` which
/// natively supports array element extraction. For runtime indices, use
/// `MirExtractArrayElementOp` instead.
#[pliron_op(
    name = "mir.extract_field",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (index: FieldIndexAttr)
)]
pub struct MirExtractFieldOp;

impl MirExtractFieldOp {
    /// Create a new MirExtractFieldOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirExtractFieldOp { op }
    }
}

impl Verify for MirExtractFieldOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let operand = op.get_operand(0);
        let operand_ty = operand.get_type(ctx);
        let operand_ty_obj = operand_ty.deref(ctx);

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);

        let index = match self.get_attr_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => return verify_err!(op.loc(), "MirExtractFieldOp missing index attribute"),
        };

        if let Some(tuple_ty) = operand_ty_obj.downcast_ref::<MirTupleType>() {
            let types = tuple_ty.get_types();
            if index >= types.len() {
                return verify_err!(op.loc(), "MirExtractFieldOp index out of bounds for tuple");
            }
            let expected_ty = types[index];
            // Types are TypeHandle, can compare directly
            if res_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp result type mismatch. Expected: {}, Actual: {}",
                    expected_ty.disp(ctx),
                    res_ty.disp(ctx)
                );
            }
        } else if let Some(slice_ty) = operand_ty_obj.downcast_ref::<MirSliceType>() {
            if index == 0 {
                // Field 0: *T (ptr to element)
                let res_ty_obj = res_ty.deref(ctx);
                if let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() {
                    if ptr_ty.pointee != slice_ty.element_ty {
                        return verify_err!(
                            op.loc(),
                            "MirExtractFieldOp result type mismatch for slice ptr"
                        );
                    }
                } else {
                    return verify_err!(
                        op.loc(),
                        "MirExtractFieldOp result must be ptr for slice field 0"
                    );
                }
            } else if index == 1 {
                // Field 1: usize (len)
                let res_ty_obj = res_ty.deref(ctx);
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        op.loc(),
                        "MirExtractFieldOp result must be integer for slice len"
                    );
                }
            } else {
                return verify_err!(op.loc(), "MirExtractFieldOp index out of bounds for slice");
            }
        } else if let Some(slice_ty) = operand_ty_obj.downcast_ref::<MirDisjointSliceType>() {
            if index == 0 {
                // Field 0: *T (ptr to element, likely mutable for disjoint slice)
                let res_ty_obj = res_ty.deref(ctx);
                if let Some(ptr_ty) = res_ty_obj.downcast_ref::<MirPtrType>() {
                    if ptr_ty.pointee != slice_ty.element_ty {
                        return verify_err!(
                            op.loc(),
                            "MirExtractFieldOp result type mismatch for disjoint slice ptr. Expected pointee: {}, Actual: {}",
                            slice_ty.element_ty.disp(ctx),
                            ptr_ty.pointee.disp(ctx)
                        );
                    }
                } else {
                    return verify_err!(
                        op.loc(),
                        "MirExtractFieldOp result must be ptr for disjoint slice field 0"
                    );
                }
            } else if index == 1 {
                // Field 1: usize (len)
                let res_ty_obj = res_ty.deref(ctx);
                if res_ty_obj.downcast_ref::<IntegerType>().is_none() {
                    return verify_err!(
                        op.loc(),
                        "MirExtractFieldOp result must be integer for disjoint slice len"
                    );
                }
            } else {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp index out of bounds for disjoint slice"
                );
            }
        } else if let Some(struct_ty) = operand_ty_obj.downcast_ref::<MirStructType>() {
            // Struct field extraction
            let field_count = struct_ty.field_count();
            if index >= field_count {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp index {} out of bounds for struct '{}' with {} fields",
                    index,
                    struct_ty.name(),
                    field_count
                );
            }
            let expected_ty = struct_ty.get_field_type(index).unwrap();
            if res_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp result type mismatch for struct field '{}'. Expected: {}, Actual: {}",
                    struct_ty.field_names()[index],
                    expected_ty.disp(ctx),
                    res_ty.disp(ctx)
                );
            }
        } else if let Some(union_ty) = operand_ty_obj.downcast_ref::<MirUnionType>() {
            let field_count = union_ty.field_count();
            if index >= field_count {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp index {} out of bounds for union '{}' with {} fields",
                    index,
                    union_ty.name(),
                    field_count
                );
            }
            let expected_ty = union_ty.get_field_type(index).unwrap();
            if res_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp result type mismatch for union field '{}'. Expected: {}, Actual: {}",
                    union_ty.field_names()[index],
                    expected_ty.disp(ctx),
                    res_ty.disp(ctx)
                );
            }
        } else if let Some(array_ty) = operand_ty_obj.downcast_ref::<MirArrayType>() {
            // Array element extraction with constant index
            // This works because LLVM's extractvalue supports arrays with constant indices
            let array_size = array_ty.size() as usize;
            if index >= array_size {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp index {} out of bounds for array of size {}",
                    index,
                    array_size
                );
            }
            let expected_ty = array_ty.element_type();
            if res_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp result type mismatch for array element. Expected: {}, Actual: {}",
                    expected_ty.disp(ctx),
                    res_ty.disp(ctx)
                );
            }
        } else if operand_ty_obj.downcast_ref::<IntegerType>().is_some() {
            // Scalar-lowered newtype case: extracting field 0 from a scalar
            // This happens with newtypes like ThreadIndex(usize)
            // The extraction is a no-op - just verify types match
            if index != 0 {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp on scalar only supports field 0 (scalar-lowered newtype)"
                );
            }
            if operand_ty != res_ty {
                return verify_err!(
                    op.loc(),
                    "MirExtractFieldOp on scalar (scalar-lowered newtype) must preserve type"
                );
            }
        } else {
            return verify_err!(
                op.loc(),
                "MirExtractFieldOp operand must be tuple, slice, struct, array, or scalar (newtype)"
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirInsertFieldOp
// ============================================================================

/// MIR insert field operation.
///
/// Inserts a value into a field of an aggregate (struct or tuple), producing a new aggregate.
///
/// # Operands
///
/// ```text
/// | Index | Name        | Description                    |
/// |-------|-------------|--------------------------------|
/// | 0     | `aggregate` | The aggregate value            |
/// | 1     | `new_value` | The new field value to insert  |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name           | Type           | Description              |
/// |----------------|----------------|--------------------------|
/// | `insert_index` | FieldIndexAttr | Index of field to update |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                              |
/// |-------|-----------------------------------|
/// | `res` | Same type as input aggregate      |
/// ```
///
/// # Verification
///
/// - Operand 0 must be a struct or tuple type.
/// - Index must be within bounds.
/// - Operand 1 type must match the field type at the given index.
/// - Result type must equal operand 0 type.
#[pliron_op(
    name = "mir.insert_field",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (insert_index: FieldIndexAttr)
)]
pub struct MirInsertFieldOp;

impl MirInsertFieldOp {
    /// Create a new MirInsertFieldOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirInsertFieldOp { op }
    }
}

impl Verify for MirInsertFieldOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let aggregate = op.get_operand(0);
        let aggregate_ty = aggregate.get_type(ctx);
        let aggregate_ty_obj = aggregate_ty.deref(ctx);

        let new_value = op.get_operand(1);
        let new_value_ty = new_value.get_type(ctx);

        let res = op.get_result(0);
        let res_ty = res.get_type(ctx);

        // Result type must match aggregate type
        if res_ty != aggregate_ty {
            return verify_err!(
                op.loc(),
                "MirInsertFieldOp result type must match aggregate type"
            );
        }

        let index = match self.get_attr_insert_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => {
                return verify_err!(op.loc(), "MirInsertFieldOp missing insert_index attribute");
            }
        };

        if let Some(tuple_ty) = aggregate_ty_obj.downcast_ref::<MirTupleType>() {
            let types = tuple_ty.get_types();
            if index >= types.len() {
                return verify_err!(op.loc(), "MirInsertFieldOp index out of bounds for tuple");
            }
            let expected_ty = types[index];
            if new_value_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp field type mismatch. Expected: {}, Actual: {}",
                    expected_ty.disp(ctx),
                    new_value_ty.disp(ctx)
                );
            }
        } else if let Some(struct_ty) = aggregate_ty_obj.downcast_ref::<MirStructType>() {
            let field_count = struct_ty.field_count();
            if index >= field_count {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp index {} out of bounds for struct '{}' with {} fields",
                    index,
                    struct_ty.name(),
                    field_count
                );
            }
            let expected_ty = struct_ty.get_field_type(index).unwrap();
            if new_value_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp field type mismatch for struct field '{}'. Expected: {}, Actual: {}",
                    struct_ty.field_names()[index],
                    expected_ty.disp(ctx),
                    new_value_ty.disp(ctx)
                );
            }
        } else if let Some(union_ty) = aggregate_ty_obj.downcast_ref::<MirUnionType>() {
            let field_count = union_ty.field_count();
            if index >= field_count {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp index {} out of bounds for union '{}' with {} fields",
                    index,
                    union_ty.name(),
                    field_count
                );
            }
            let expected_ty = union_ty.get_field_type(index).unwrap();
            if new_value_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp field type mismatch for union field '{}'. Expected: {}, Actual: {}",
                    union_ty.field_names()[index],
                    expected_ty.disp(ctx),
                    new_value_ty.disp(ctx)
                );
            }
        } else if let Some(array_ty) = aggregate_ty_obj.downcast_ref::<MirArrayType>() {
            // Array support: arr[i] = value with constant index
            let array_len = array_ty.size() as usize;
            if index >= array_len {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp index {} out of bounds for array of length {}",
                    index,
                    array_len
                );
            }
            let expected_ty = array_ty.element_type();
            if new_value_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirInsertFieldOp element type mismatch for array. Expected: {}, Actual: {}",
                    expected_ty.disp(ctx),
                    new_value_ty.disp(ctx)
                );
            }
        } else {
            return verify_err!(
                op.loc(),
                "MirInsertFieldOp aggregate operand must be tuple, struct, or array"
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirConstructStructOp
// ============================================================================

/// MIR construct struct operation.
///
/// Constructs a struct value from individual field values.
///
/// # Operands
///
/// Takes N operands (one per field), in field order.
///
/// # Results
///
/// ```text
/// | Name  | Type          |
/// |-------|---------------|
/// | `res` | MirStructType |
/// ```
///
/// # Verification
///
/// - Number of operands must equal struct field count.
/// - Each operand type must match corresponding field type.
/// - Result type must be a struct type.
#[pliron_op(
    name = "mir.construct_struct",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface]
)]
pub struct MirConstructStructOp;

impl MirConstructStructOp {
    /// Create a new MirConstructStructOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstructStructOp { op }
    }
}

impl Verify for MirConstructStructOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Result must be a struct type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let struct_ty = match result_ty_obj.downcast_ref::<MirStructType>() {
            Some(st) => st,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirConstructStructOp result must be a struct type"
                );
            }
        };

        // Verify operand count matches field count
        let num_operands = op.get_num_operands();
        let num_fields = struct_ty.field_count();
        if num_operands != num_fields {
            return verify_err!(
                op.loc(),
                "MirConstructStructOp has {} operands but struct '{}' has {} fields",
                num_operands,
                struct_ty.name(),
                num_fields
            );
        }

        // Verify each operand type matches field type
        for i in 0..num_fields {
            let operand = op.get_operand(i);
            let operand_ty = operand.get_type(ctx);
            let expected_ty = struct_ty.get_field_type(i).unwrap();

            if operand_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirConstructStructOp operand {} type mismatch for field '{}'. Expected: {}, Actual: {}",
                    i,
                    struct_ty.field_names()[i],
                    expected_ty.disp(ctx),
                    operand_ty.disp(ctx)
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// MirConstructTupleOp
// ============================================================================

/// MIR construct tuple operation.
///
/// Constructs a tuple value from individual element values.
///
/// # Operands
///
/// Takes N operands (one per element), in element order.
///
/// # Results
///
/// ```text
/// | Name  | Type         |
/// |-------|--------------|
/// | `res` | MirTupleType |
/// ```
///
/// # Verification
///
/// - Number of operands must equal tuple element count.
/// - Each operand type must match corresponding element type.
/// - Result type must be a tuple type.
#[pliron_op(
    name = "mir.construct_tuple",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface]
)]
pub struct MirConstructTupleOp;

impl MirConstructTupleOp {
    /// Create a new MirConstructTupleOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstructTupleOp { op }
    }
}

impl Verify for MirConstructTupleOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Result must be a tuple type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let tuple_ty = match result_ty_obj.downcast_ref::<MirTupleType>() {
            Some(tt) => tt,
            None => {
                return verify_err!(op.loc(), "MirConstructTupleOp result must be a tuple type");
            }
        };

        // Verify operand count matches element count
        let num_operands = op.get_num_operands();
        let element_types = tuple_ty.get_types();
        let num_elements = element_types.len();
        if num_operands != num_elements {
            return verify_err!(
                op.loc(),
                "MirConstructTupleOp has {} operands but tuple has {} elements",
                num_operands,
                num_elements
            );
        }

        // Verify each operand type matches element type
        for (i, &expected_ty) in element_types.iter().enumerate().take(num_elements) {
            let operand = op.get_operand(i);
            let operand_ty = operand.get_type(ctx);

            if operand_ty != expected_ty {
                return verify_err!(
                    op.loc(),
                    "MirConstructTupleOp operand {} type mismatch. Expected: {}, Actual: {}",
                    i,
                    expected_ty.disp(ctx),
                    operand_ty.disp(ctx)
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// MirConstructSliceOp
// ============================================================================

/// MIR construct slice operation.
///
/// Constructs a slice fat pointer (`&[T]` / `*const [T]` / `*mut [T]`) from
/// a data pointer and a length.
///
/// # Why This Op Exists
///
/// Re-slicing in Rust (`&bytes[2..]`) goes through core's
/// `slice::index::get_offset_len_noubcheck`, which calls the
/// `aggregate_raw_ptr` intrinsic. Rustc lowers that intrinsic to
/// `Rvalue::Aggregate(AggregateKind::RawPtr(..), [data_ptr, len])` in MIR.
/// The same MIR shape is produced by `ptr::slice_from_raw_parts` and
/// `ptr::from_raw_parts`. This op represents that fat-pointer construction
/// in `dialect-mir`.
///
/// # Example
///
/// ```text
/// Rust:         let tail: &[u8] = &bytes[2..];
/// Rust MIR:     _tail = *const [u8] from (_ptr, _len)
/// dialect-mir:  %tail = mir.construct_slice (%ptr, %len) : mir.slice<u8>
/// ```
///
/// # Operands
///
/// ```text
/// | Index | Name   | Type                       | Description              |
/// |-------|--------|----------------------------|--------------------------|
/// | 0     | `ptr`  | MirPtrType<T>              | Data pointer             |
/// | 1     | `len`  | Integer type (usize)       | Number of elements       |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type            |
/// |-------|-----------------|
/// | `res` | MirSliceType<T> |
/// ```
///
/// # LLVM Lowering
///
/// `MirSliceType` lowers to the fat-pointer struct `{ ptr, i64 }`, so this
/// op becomes `llvm.undef` + two `llvm.insertvalue` operations:
/// ```text
/// %u  = llvm.undef : { ptr, i64 }
/// %t  = llvm.insertvalue %u, %ptr, [0]
/// %s  = llvm.insertvalue %t, %len, [1]
/// ```
///
/// # Verification
///
/// - Result type must be a slice type.
/// - Operand 0 must be a pointer whose pointee is the slice element type.
/// - Operand 1 must be an integer type.
#[pliron_op(
    name = "mir.construct_slice",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirConstructSliceOp;

impl MirConstructSliceOp {
    /// Create a new MirConstructSliceOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstructSliceOp { op }
    }
}

impl Verify for MirConstructSliceOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Result must be a slice type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let slice_ty = match result_ty_obj.downcast_ref::<MirSliceType>() {
            Some(st) => st,
            None => {
                return verify_err!(op.loc(), "MirConstructSliceOp result must be a slice type");
            }
        };

        // Operand 0 must be a pointer to the slice element type
        let ptr_operand = op.get_operand(0);
        let ptr_ty = ptr_operand.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);
        match ptr_ty_obj.downcast_ref::<MirPtrType>() {
            Some(ptr_ty) => {
                if ptr_ty.pointee != slice_ty.element_ty {
                    return verify_err!(
                        op.loc(),
                        "MirConstructSliceOp data pointer pointee mismatch. Expected: {}, Actual: {}",
                        slice_ty.element_ty.disp(ctx),
                        ptr_ty.pointee.disp(ctx)
                    );
                }
            }
            None => {
                return verify_err!(
                    op.loc(),
                    "MirConstructSliceOp operand 0 must be a pointer type, got: {}",
                    ptr_ty.disp(ctx)
                );
            }
        }

        // Operand 1 must be an integer type (the length)
        let len_operand = op.get_operand(1);
        let len_ty = len_operand.get_type(ctx);
        let len_ty_obj = len_ty.deref(ctx);
        if len_ty_obj.downcast_ref::<IntegerType>().is_none() {
            return verify_err!(
                op.loc(),
                "MirConstructSliceOp operand 1 (length) must be an integer type, got: {}",
                len_ty.disp(ctx)
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirConstructArrayOp
// ============================================================================

/// MIR construct array operation.
///
/// Constructs an array value from individual element values.
///
/// # Why This Op Exists
///
/// Rust array literals like `[1.0, 2.0, 3.0, 4.0]` compile to
/// `AggregateKind::Array` in rustc's MIR. This op represents that construct
/// in `dialect-mir`, before lowering to the LLVM dialect.
///
/// # Example
///
/// ```text
/// Rust:         let arr: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
/// Rust MIR:     _arr = [const 1.0f32, const 2.0f32, const 3.0f32, const 4.0f32]
/// dialect-mir:  %arr = mir.construct_array (%c1, %c2, %c3, %c4) : mir.array<4 x f32>
/// ```
///
/// # Operands
///
/// Takes N operands (one per element), in element order.
/// All operands must have the same type.
///
/// # Results
///
/// ```text
/// | Name  | Type         |
/// |-------|--------------|
/// | `res` | MirArrayType |
/// ```
///
/// # LLVM Lowering
///
/// Lowered to a chain of `llvm.insertvalue` operations starting from `undef`:
/// ```text
/// %t1  = llvm.insertvalue undef, %c1, [0]
/// %t2  = llvm.insertvalue %t1,   %c2, [1]
/// %arr = llvm.insertvalue %t2,   %c3, [2]
/// ...
/// ```
///
/// # Verification
///
/// - Number of operands must equal array size.
/// - All operand types must match the element type.
/// - Result type must be an array type.
#[pliron_op(
    name = "mir.construct_array",
    format,
    interfaces = [NResultsInterface<1>, OneResultInterface]
)]
pub struct MirConstructArrayOp;

impl MirConstructArrayOp {
    /// Create a new MirConstructArrayOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirConstructArrayOp { op }
    }
}

impl Verify for MirConstructArrayOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Result must be an array type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let array_ty = match result_ty_obj.downcast_ref::<MirArrayType>() {
            Some(at) => at,
            None => {
                return verify_err!(op.loc(), "MirConstructArrayOp result must be an array type");
            }
        };

        // Verify operand count matches array size
        let num_operands = op.get_num_operands();
        let array_size = array_ty.size() as usize;
        if num_operands != array_size {
            return verify_err!(
                op.loc(),
                "MirConstructArrayOp has {} operands but array has {} elements",
                num_operands,
                array_size
            );
        }

        // Verify each operand type matches element type
        let element_ty = array_ty.element_type();
        for i in 0..array_size {
            let operand = op.get_operand(i);
            let operand_ty = operand.get_type(ctx);

            if operand_ty != element_ty {
                return verify_err!(
                    op.loc(),
                    "MirConstructArrayOp operand {} type mismatch. Expected: {}, Actual: {}",
                    i,
                    element_ty.disp(ctx),
                    operand_ty.disp(ctx)
                );
            }
        }

        Ok(())
    }
}

// ============================================================================
// MirExtractArrayElementOp
// ============================================================================

/// MIR extract array element operation with runtime index.
///
/// Extracts an element from an array using a runtime index value.
/// This is different from `MirExtractFieldOp` which uses a compile-time constant index.
///
/// # Why This Op Exists
///
/// LLVM's `extractvalue` instruction only accepts **constant** indices:
/// ```llvm
/// %val = extractvalue [4 x float] %arr, 0      ; ✓ Constant index works
/// %val = extractvalue [4 x float] %arr, %idx   ; ✗ ILLEGAL - runtime index
/// ```
///
/// When Rust code indexes an array with a runtime value (e.g., `arr[tid % 4]`),
/// we need this op to represent that access before lowering to the
/// alloca→store→GEP→load pattern that LLVM requires.
///
/// # Example
///
/// ```text
/// Rust:        let val = arr[thread_id % 4];   // thread_id is runtime
/// rustc MIR:   _idx = Rem(_tid, 4_usize)
///              _val = _arr[_idx]               // ProjectionElem::Index
/// Our MIR:     %val = mir.extract_array_element %arr, %idx : f32
/// ```
///
/// # Operands
///
/// ```text
/// | Index | Name    | Type          |
/// |-------|---------|---------------|
/// | 0     | `array` | MirArrayType  |
/// | 1     | `index` | Integer type  |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type                |
/// |-------|---------------------|
/// | `res` | Array element type  |
/// ```
///
/// # LLVM Lowering
///
/// Since LLVM's `extractvalue` only supports constant indices, this op
/// is lowered to:
/// ```text
/// %ptr      = llvm.alloca [4 x float]       ; 1. Stack allocate
/// llvm.store %arr, %ptr                     ; 2. Store array to memory
/// %elem_ptr = llvm.gep %ptr, [0, %idx]      ; 3. Compute element address
/// %val      = llvm.load %elem_ptr           ; 4. Load element
/// ```
#[pliron_op(
    name = "mir.extract_array_element",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirExtractArrayElementOp;

impl MirExtractArrayElementOp {
    /// Create a new MirExtractArrayElementOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirExtractArrayElementOp { op }
    }
}

impl Verify for MirExtractArrayElementOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let array_operand = op.get_operand(0);
        let array_ty = array_operand.get_type(ctx);
        let array_ty_obj = array_ty.deref(ctx);

        // First operand must be an array type
        let element_ty = match array_ty_obj.downcast_ref::<MirArrayType>() {
            Some(at) => at.element_type(),
            None => {
                return verify_err!(
                    op.loc(),
                    "MirExtractArrayElementOp first operand must be an array type"
                );
            }
        };

        // Second operand must be an integer type (the index)
        let index_operand = op.get_operand(1);
        let index_ty = index_operand.get_type(ctx);
        let index_ty_obj = index_ty.deref(ctx);
        if index_ty_obj.downcast_ref::<IntegerType>().is_none() {
            return verify_err!(
                op.loc(),
                "MirExtractArrayElementOp second operand must be an integer type"
            );
        }

        // Result type must match array element type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        if result_ty != element_ty {
            return verify_err!(
                op.loc(),
                "MirExtractArrayElementOp result type must match array element type"
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirFieldAddrOp
// ============================================================================

/// MIR field address operation.
///
/// Computes the address of a struct field from a pointer to the struct.
/// This is the address-of-field operation needed for mutable references
/// to nested struct fields.
///
/// # Why This Exists
///
/// When Rust code takes `&mut self.field`, we need the ADDRESS of the field,
/// not a COPY of its value. Consider `Enumerate::next()`:
///
/// ```text
/// MIR: _5 = &mut ((*_1).0: I)   // Take reference to field 0 of struct at *_1
///
/// WRONG (what mir.ref does):
///   load struct from _1          → struct_value
///   extract_field(struct_value, 0) → field_value
///   mir.ref(field_value)         → new_ptr (COPY on stack!)
///
/// CORRECT (what mir.field_addr does):
///   mir.field_addr(_1, 0)        → field_ptr (address INTO original struct)
/// ```
///
/// Without this, mutations through the reference don't affect the original struct,
/// causing bugs like infinite loops in iterators (the iterator state never advances).
///
/// # Operands
///
/// ```text
/// | Name      | Type       | Description                    |
/// |-----------|------------|--------------------------------|
/// | `ptr`     | MirPtrType | Pointer to a struct            |
/// ```
///
/// # Attributes
///
/// ```text
/// | Name          | Type           | Description              |
/// |---------------|----------------|--------------------------|
/// | `field_index` | FieldIndexAttr | Index of field to access |
/// ```
///
/// # Results
///
/// ```text
/// | Name  | Type       | Description                      |
/// |-------|------------|----------------------------------|
/// | `res` | MirPtrType | Pointer to the field             |
/// ```
///
/// # Lowering
///
/// Lowers to LLVM GEP (getelementptr) with indices [0, field_index]:
/// ```llvm
/// %field_ptr = getelementptr inbounds %StructType, ptr %struct_ptr, i32 0, i32 <field_index>
/// ```
#[pliron_op(
    name = "mir.field_addr",
    format,
    interfaces = [NOpdsInterface<1>, OneOpdInterface, NResultsInterface<1>, OneResultInterface],
    attributes = (field_index: FieldIndexAttr)
)]
pub struct MirFieldAddrOp;

impl MirFieldAddrOp {
    /// Create a new MirFieldAddrOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirFieldAddrOp { op }
    }
}

impl Verify for MirFieldAddrOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // Operand must be a pointer type
        let ptr_operand = op.get_operand(0);
        let ptr_ty = ptr_operand.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);

        let ptr_type = match ptr_ty_obj.downcast_ref::<MirPtrType>() {
            Some(p) => p,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirFieldAddrOp operand must be a pointer type, got: {}",
                    ptr_ty.disp(ctx)
                );
            }
        };

        // Pointee must be a struct or union type.
        let pointee_ty = ptr_type.pointee;
        let pointee_ty_obj = pointee_ty.deref(ctx);

        let index = match self.get_attr_field_index(ctx) {
            Some(attr) => attr.0 as usize,
            None => return verify_err!(op.loc(), "MirFieldAddrOp missing field_index attribute"),
        };

        let field_types = if let Some(struct_ty) = pointee_ty_obj.downcast_ref::<MirStructType>() {
            struct_ty.field_types()
        } else if let Some(union_ty) = pointee_ty_obj.downcast_ref::<MirUnionType>() {
            union_ty.field_types()
        } else {
            return verify_err!(
                op.loc(),
                "MirFieldAddrOp pointer must point to a struct or union type, got: {}",
                pointee_ty.disp(ctx)
            );
        };

        // Index must be valid.
        if index >= field_types.len() {
            return verify_err!(
                op.loc(),
                "MirFieldAddrOp field_index {} out of bounds for struct with {} fields",
                index,
                field_types.len()
            );
        }

        // Result must be a pointer to the field type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let result_ptr_ty = match result_ty_obj.downcast_ref::<MirPtrType>() {
            Some(p) => p,
            None => {
                return verify_err!(op.loc(), "MirFieldAddrOp result must be a pointer type");
            }
        };

        let expected_field_ty = field_types[index];
        if result_ptr_ty.pointee != expected_field_ty {
            return verify_err!(
                op.loc(),
                "MirFieldAddrOp result pointer type mismatch. Expected pointer to: {}, got pointer to: {}",
                expected_field_ty.disp(ctx),
                result_ptr_ty.pointee.disp(ctx)
            );
        }

        Ok(())
    }
}

// ============================================================================
// MirArrayElementAddrOp
// ============================================================================

/// MIR array element address operation.
///
/// Computes the address of an array element from a pointer to the array
/// and a runtime index value. This is the array analog of `MirFieldAddrOp`.
///
/// # Why This Exists
///
/// When arrays need runtime index access (read or write), we need to work
/// with memory locations rather than SSA values. This op computes the
/// address of an element:
///
/// ```text
/// Rust:         arr[i] = val;  // or: let x = arr[i];
///
/// dialect-mir:
///   %elem_ptr = mir.array_element_addr %arr_ptr, %i : !mir.ptr<[16 x f32]>, i64 -> !mir.ptr<f32>
///   mir.store %val, %elem_ptr      // for writes
///   %x = mir.load %elem_ptr        // for reads
/// ```
///
/// This enables O(1) element access without copying the entire array.
///
/// # Operands
///
/// ```text
/// | Index | Name      | Type                 | Description                |
/// |-------|-----------|----------------------|----------------------------|
/// | 0     | `arr_ptr` | MirPtrType<[N x T]>  | Pointer to an array        |
/// | 1     | `index`   | Integer type         | Runtime index value        |
/// ```
///
/// # Results
///
/// ```text
/// | Name       | Type           | Description                    |
/// |------------|----------------|--------------------------------|
/// | `elem_ptr` | MirPtrType<T>  | Pointer to the indexed element |
/// ```
///
/// # LLVM Lowering
///
/// Lowers to a single LLVM GEP instruction:
/// ```llvm
/// %elem_ptr = getelementptr [N x T], ptr %arr_ptr, i64 0, i64 %index
/// ```
#[pliron_op(
    name = "mir.array_element_addr",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface]
)]
pub struct MirArrayElementAddrOp;

impl MirArrayElementAddrOp {
    /// Create a new MirArrayElementAddrOp wrapper.
    pub fn new(op: Ptr<Operation>) -> Self {
        MirArrayElementAddrOp { op }
    }
}

impl Verify for MirArrayElementAddrOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        // First operand must be a pointer to an array type
        let ptr_operand = op.get_operand(0);
        let ptr_ty = ptr_operand.get_type(ctx);
        let ptr_ty_obj = ptr_ty.deref(ctx);

        let ptr_type = match ptr_ty_obj.downcast_ref::<MirPtrType>() {
            Some(p) => p,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirArrayElementAddrOp first operand must be a pointer type, got: {}",
                    ptr_ty.disp(ctx)
                );
            }
        };

        // Pointee must be an array type
        let pointee_ty = ptr_type.pointee;
        let pointee_ty_obj = pointee_ty.deref(ctx);
        let array_ty = match pointee_ty_obj.downcast_ref::<MirArrayType>() {
            Some(a) => a,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirArrayElementAddrOp pointer must point to an array type, got: {}",
                    pointee_ty.disp(ctx)
                );
            }
        };

        // Second operand must be an integer type (the index)
        let index_operand = op.get_operand(1);
        let index_ty = index_operand.get_type(ctx);
        let index_ty_obj = index_ty.deref(ctx);
        if index_ty_obj.downcast_ref::<IntegerType>().is_none() {
            return verify_err!(
                op.loc(),
                "MirArrayElementAddrOp second operand must be an integer type, got: {}",
                index_ty.disp(ctx)
            );
        }

        // Result must be a pointer to the element type
        let result = op.get_result(0);
        let result_ty = result.get_type(ctx);
        let result_ty_obj = result_ty.deref(ctx);

        let result_ptr_ty = match result_ty_obj.downcast_ref::<MirPtrType>() {
            Some(p) => p,
            None => {
                return verify_err!(
                    op.loc(),
                    "MirArrayElementAddrOp result must be a pointer type"
                );
            }
        };

        let expected_elem_ty = array_ty.element_type();
        if result_ptr_ty.pointee != expected_elem_ty {
            return verify_err!(
                op.loc(),
                "MirArrayElementAddrOp result pointer type mismatch. Expected pointer to: {}, got pointer to: {}",
                expected_elem_ty.disp(ctx),
                result_ptr_ty.pointee.disp(ctx)
            );
        }

        // Address space must match
        if result_ptr_ty.address_space != ptr_type.address_space {
            return verify_err!(
                op.loc(),
                "MirArrayElementAddrOp address space mismatch. Expected: {:?}, got: {:?}",
                ptr_type.address_space,
                result_ptr_ty.address_space
            );
        }

        Ok(())
    }
}

/// Register aggregate operations into the given context.
pub fn register(ctx: &mut Context) {
    MirExtractFieldOp::register(ctx);
    MirInsertFieldOp::register(ctx);
    MirConstructStructOp::register(ctx);
    MirConstructTupleOp::register(ctx);
    MirConstructSliceOp::register(ctx);
    MirConstructArrayOp::register(ctx);
    MirExtractArrayElementOp::register(ctx);
    MirFieldAddrOp::register(ctx);
    MirArrayElementAddrOp::register(ctx);
}
