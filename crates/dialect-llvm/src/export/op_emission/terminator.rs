/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    basic_block::BasicBlock, context::Ptr, operation::Operation, r#type::Typed, value::Value,
};

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_return_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        write!(output, "  ret ").unwrap();
        if op_ref.operands().count() == 0 {
            write!(output, "void").unwrap();
        } else {
            let val = op_ref.operands().next().unwrap();
            self.export_type(val.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(val, value_names, output)?;
        }
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_unreachable_op(&self, output: &mut String) {
        writeln!(output, "  unreachable").unwrap();
    }

    pub(super) fn export_br_op(
        &self,
        op_ref: &Operation,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let dest = op_ref.successors().next().unwrap();
        let label = block_labels.get(&dest).ok_or("Missing block label")?;
        writeln!(output, "  br label %{label}").unwrap();
        Ok(())
    }

    pub(super) fn export_cond_br_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        block_labels: &HashMap<Ptr<BasicBlock>, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let mut succs = op_ref.successors();
        let true_dest = succs.next().unwrap();
        let false_dest = succs.next().unwrap();
        let true_label = block_labels.get(&true_dest).ok_or("Missing true label")?;
        let false_label = block_labels.get(&false_dest).ok_or("Missing false label")?;
        let cond = op_ref.get_operand(0);

        write!(output, "  br i1 ").unwrap();
        self.export_value(cond, value_names, output)?;
        writeln!(output, ", label %{true_label}, label %{false_label}").unwrap();
        Ok(())
    }
}
