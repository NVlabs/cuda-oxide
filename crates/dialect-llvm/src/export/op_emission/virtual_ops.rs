/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;

use pliron::{
    builtin::attributes::{FPDoubleAttr, FPSingleAttr, IntegerAttr},
    operation::Operation,
    value::Value,
};

use crate::{attributes::FPHalfAttr, ops};

use super::super::{ModuleExportState, format_float_literal, format_half_literal};

impl<'a> ModuleExportState<'a> {
    pub(super) fn check_address_of_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
    ) {
        // AddressOfOp is virtual in textual LLVM IR: the naming pre-pass
        // registers its result as the global symbol printed at use sites.
        let res = op_ref.get_result(0);
        debug_assert!(
            value_names
                .get(&res)
                .is_some_and(|name| name.starts_with('@')),
            "AddressOfOp result must be pre-registered as a global \
             symbol by the naming pre-pass; got {:?}",
            value_names.get(&res),
        );
    }

    pub(super) fn export_constant_op(
        &self,
        op_ref: &Operation,
        const_op: &ops::ConstantOp,
        value_names: &mut HashMap<Value, String>,
    ) {
        let val_attr = const_op.get_value(self.ctx);
        let const_str = if let Some(int_attr) = val_attr.downcast_ref::<IntegerAttr>() {
            // Use APInt's decimal conversion instead of parsing debug output,
            // which may contain underscore-grouped hex values.
            int_attr.value().to_string_unsigned_decimal()
        } else if let Some(fp16_attr) = val_attr.downcast_ref::<FPHalfAttr>() {
            format_half_literal(fp16_attr.to_bits())
        } else if let Some(fp32_attr) = val_attr.downcast_ref::<FPSingleAttr>() {
            let float_val: f32 = fp32_attr.clone().into();
            format_float_literal(f64::from(float_val))
        } else if let Some(fp64_attr) = val_attr.downcast_ref::<FPDoubleAttr>() {
            let float_val: f64 = fp64_attr.clone().into();
            format_float_literal(float_val)
        } else {
            "0".to_string()
        };

        let res = op_ref.get_result(0);
        value_names.insert(res, const_str);
    }
}
