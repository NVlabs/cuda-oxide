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
    pub(super) fn export_inline_asm_op(
        &mut self,
        op_ref: &Operation,
        inline_asm: &ops::InlineAsmOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let asm_template = inline_asm.asm_template(self.ctx);
        let constraints = inline_asm.constraints(self.ctx);
        let is_convergent = inline_asm.is_convergent(self.ctx);

        if op_ref.get_num_results() > 0 {
            let res = op_ref.get_result(0);
            let res_name = value_names.get(&res).unwrap();
            let res_ty = res.get_type(self.ctx);
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(res_ty, output)?;
        } else {
            write!(output, "  call void").unwrap();
        }

        write!(
            output,
            " asm sideeffect \"{asm_template}\", \"{constraints}\"("
        )
        .unwrap();

        self.export_inline_asm_args(op_ref, value_names, output)?;
        self.finish_inline_asm_call(is_convergent, output);
        Ok(())
    }

    pub(super) fn export_inline_asm_multi_op(
        &mut self,
        op_ref: &Operation,
        inline_asm: &ops::InlineAsmMultiOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let asm_template = inline_asm.asm_template(self.ctx);
        let constraints = inline_asm.constraints(self.ctx);
        let is_convergent = inline_asm.is_convergent(self.ctx);
        let num_results = op_ref.get_num_results();

        if num_results == 0 {
            write!(output, "  call void").unwrap();
            write!(
                output,
                " asm sideeffect \"{asm_template}\", \"{constraints}\"("
            )
            .unwrap();

            self.export_inline_asm_args(op_ref, value_names, output)?;
            self.finish_inline_asm_call(is_convergent, output);
            return Ok(());
        }

        let mut struct_type = String::from("{");
        for i in 0..num_results {
            if i > 0 {
                struct_type.push_str(", ");
            }
            let res_ty = op_ref.get_result(i).get_type(self.ctx);
            let mut ty_str = String::new();
            self.export_type(res_ty, &mut ty_str)?;
            struct_type.push_str(&ty_str);
        }
        struct_type.push('}');

        let first_res = op_ref.get_result(0);
        let first_res_name = value_names.get(&first_res).unwrap();
        let struct_result_name = format!("{first_res_name}_struct");

        write!(output, "  {struct_result_name} = call {struct_type}").unwrap();
        write!(
            output,
            " asm sideeffect \"{asm_template}\", \"{constraints}\"("
        )
        .unwrap();

        self.export_inline_asm_args(op_ref, value_names, output)?;
        self.finish_inline_asm_call(is_convergent, output);

        for i in 0..num_results {
            let res = op_ref.get_result(i);
            let res_name = value_names.get(&res).unwrap();

            writeln!(
                output,
                "  {res_name} = extractvalue {struct_type} {struct_result_name}, {i}"
            )
            .unwrap();
        }

        Ok(())
    }

    fn export_inline_asm_args(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        Ok(())
    }

    fn finish_inline_asm_call(&mut self, is_convergent: bool, output: &mut String) {
        if is_convergent {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }
    }
}
