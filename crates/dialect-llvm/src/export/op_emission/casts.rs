/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{context::Ptr, op::Op, operation::Operation, r#type::Typed, value::Value};

use crate::ops;

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_cast(
        &self,
        op_name: &str,
        op: Ptr<Operation>,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let op_ref = op.deref(self.ctx);
        let res = op_ref.get_result(0);
        let val = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();

        write!(output, "  {res_name} = {op_name} ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_zext_op(
        &self,
        op_ref: &Operation,
        zext: &ops::ZExtOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let val = op_ref.get_operand(0);
        let nneg_key: pliron::identifier::Identifier = "llvm_nneg_flag".try_into().unwrap();
        let nneg = zext
            .get_operation()
            .deref(self.ctx)
            .attributes
            .0
            .get(&nneg_key)
            .and_then(|attr| {
                attr.downcast_ref::<pliron::builtin::attributes::BoolAttr>()
                    .map(|b| bool::from(b.clone()))
            })
            .unwrap_or(false);

        write!(output, "  {res_name} = zext ").unwrap();
        if nneg {
            write!(output, "nneg ").unwrap();
        }
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        write!(output, " to ").unwrap();
        self.export_type(res.get_type(self.ctx), output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_undef_op(
        &self,
        op_ref: &Operation,
        value_names: &mut HashMap<Value, String>,
    ) {
        let res = op_ref.get_result(0);
        value_names.insert(res, "undef".to_string());
    }
}
