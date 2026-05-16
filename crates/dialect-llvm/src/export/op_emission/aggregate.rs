/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{operation::Operation, r#type::Typed, value::Value};

use crate::ops;

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_extract_value_op(
        &self,
        op_ref: &Operation,
        extract_op: &ops::ExtractValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);
        let indices = extract_op.indices(self.ctx);

        write!(output, "  {res_name} = extractvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        for idx in indices {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_insert_value_op(
        &self,
        op_ref: &Operation,
        insert_op: &ops::InsertValueOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let agg = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);
        let indices = insert_op.indices(self.ctx);

        write!(output, "  {res_name} = insertvalue ").unwrap();
        self.export_type(agg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(agg, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;

        for idx in indices {
            write!(output, ", {idx}").unwrap();
        }
        writeln!(output).unwrap();
        Ok(())
    }
}
