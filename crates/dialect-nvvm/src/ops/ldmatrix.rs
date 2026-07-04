/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Warp-cooperative shared-memory matrix loads.
//!
//! Each operation loads one, two, or four 8×8 matrices of b16 elements and
//! returns the calling lane's packed 32-bit register fragments. The `.trans`
//! forms select column-major layout.
//!
//! # Address lanes
//!
//! ```text
//! x1: lanes  0..7  provide row addresses
//! x2: lanes  0..15 provide row addresses
//! x4: lanes  0..31 provide row addresses
//! ```
//!
//! On sm_75, every lane must still supply a valid address for x1 and x2; copy
//! the lower-lane addresses into the otherwise unused upper lanes. Each group
//! of four lanes loads 16 bytes from a naturally aligned address.
//!
//! These are weak memory operations: `.sync` makes the warp converge at the
//! instruction, but does not order earlier or later memory accesses. Callers
//! need an appropriate barrier or fence before a dependent access.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::IntegerType,
    common_traits::Verify,
    context::{Context, Ptr},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

fn verify_ldmatrix(
    ctx: &Context,
    op: Ptr<Operation>,
    op_name: &str,
    result_count: usize,
) -> Result<(), Error> {
    let op = &*op.deref(ctx);
    let operands: Vec<_> = op.operands().collect();
    if operands.len() != 1 {
        return verify_err!(op.loc(), "{} requires one shared-memory pointer", op_name);
    }

    let pointer_ty = operands[0].get_type(ctx);
    let pointer_ty = pointer_ty.deref(ctx);
    let Some(pointer_ty) = pointer_ty.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{} operand 0 must be a MIR pointer", op_name);
    };
    if !matches!(
        pointer_ty.address_space,
        address_space::GENERIC | address_space::SHARED
    ) {
        return verify_err!(
            op.loc(),
            "{} operand 0 must be a generic or shared-memory pointer, not address space {}",
            op_name,
            pointer_ty.address_space
        );
    }

    let pointee = pointer_ty.pointee.deref(ctx);
    let Some(pointee) = pointee.downcast_ref::<IntegerType>() else {
        return verify_err!(
            op.loc(),
            "{} operand 0 must point to a 32-bit integer carrier",
            op_name
        );
    };
    if pointee.width() != 32 {
        return verify_err!(
            op.loc(),
            "{} operand 0 must point to a 32-bit integer carrier",
            op_name
        );
    }

    if op.get_num_results() != result_count {
        return verify_err!(
            op.loc(),
            "{} requires {} register results",
            op_name,
            result_count
        );
    }

    for index in 0..result_count {
        let ty = op.get_result(index).get_type(ctx);
        let ty = ty.deref(ctx);
        let Some(int_ty) = ty.downcast_ref::<IntegerType>() else {
            return verify_err!(op.loc(), "{} result {} must be an integer", op_name, index);
        };
        if int_ty.width() != 32 {
            return verify_err!(op.loc(), "{} result {} must be 32 bits", op_name, index);
        }
    }

    Ok(())
}

macro_rules! define_ldmatrix_op {
    (
        $(#[$doc:meta])*
        $name:ident,
        $op_name:literal,
        $results:literal
    ) => {
        $(#[$doc])*
        #[pliron_op(
            name = $op_name,
            format,
            interfaces = [NOpdsInterface<1>, NResultsInterface<$results>],
        )]
        pub struct $name;

        impl $name {
            /// Wrap an existing operation pointer.
            pub fn new(op: Ptr<Operation>) -> Self {
                Self { op }
            }
        }

        impl Verify for $name {
            fn verify(&self, ctx: &Context) -> Result<(), Error> {
                verify_ldmatrix(ctx, self.get_operation(), $op_name, $results)
            }
        }
    };
}

define_ldmatrix_op!(
    /// Load one row-major 8×8 b16 matrix.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x1.shared.b16 {$0}, [addr];`
    LdmatrixX1Op,
    "nvvm.ldmatrix_x1",
    1
);

define_ldmatrix_op!(
    /// Load one column-major 8×8 b16 matrix.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {$0}, [addr];`
    LdmatrixX1TransOp,
    "nvvm.ldmatrix_x1_trans",
    1
);

define_ldmatrix_op!(
    /// Load two row-major 8×8 b16 matrices.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [addr];`
    LdmatrixX2Op,
    "nvvm.ldmatrix_x2",
    2
);

define_ldmatrix_op!(
    /// Load two column-major 8×8 b16 matrices.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {$0, $1}, [addr];`
    LdmatrixX2TransOp,
    "nvvm.ldmatrix_x2_trans",
    2
);

define_ldmatrix_op!(
    /// Load four row-major 8×8 b16 matrices.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [addr];`
    LdmatrixX4Op,
    "nvvm.ldmatrix_x4",
    4
);

define_ldmatrix_op!(
    /// Load four column-major 8×8 b16 matrices.
    ///
    /// PTX: `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {$0, $1, $2, $3}, [addr];`
    LdmatrixX4TransOp,
    "nvvm.ldmatrix_x4_trans",
    4
);

/// Register ldmatrix operations with the context.
pub(super) fn register(ctx: &mut Context) {
    LdmatrixX1Op::register(ctx);
    LdmatrixX1TransOp::register(ctx);
    LdmatrixX2Op::register(ctx);
    LdmatrixX2TransOp::register(ctx);
    LdmatrixX4Op::register(ctx);
    LdmatrixX4TransOp::register(ctx);
}
