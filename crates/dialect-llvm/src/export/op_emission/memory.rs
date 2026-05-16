/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{operation::Operation, r#type::Typed, value::Value};

use crate::{attributes::GepIndexAttr, ops, types::PointerType};

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_load_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        write!(output, "  {res_name} = load ").unwrap();
        self.export_type(ty, output)?;
        if addrspace != 0 {
            write!(output, ", ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, ", ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_store_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        write!(output, "  store ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        if addrspace != 0 {
            write!(output, ", ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, ", ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_alloca_op(
        &self,
        op_ref: &Operation,
        alloca_op: &ops::AllocaOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let elem_ty = alloca_op
            .get_attr_alloca_element_type(self.ctx)
            .expect("Missing alloca_element_type");
        let elem_ty_ptr = elem_ty.get_type(self.ctx);

        write!(output, "  {res_name} = alloca ").unwrap();
        self.export_type(elem_ty_ptr, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_gep_op(
        &self,
        op_ref: &Operation,
        gep_op: &ops::GetElementPtrOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let ptr = op_ref.get_operand(0);
        let elem_ty = gep_op
            .get_attr_gep_src_elem_type(self.ctx)
            .expect("Missing gep_src_elem_type")
            .get_type(self.ctx);

        write!(output, "  {res_name} = getelementptr inbounds ").unwrap();
        self.export_type(elem_ty, output)?;

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        if addrspace != 0 {
            write!(output, ", ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, ", ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;

        let indices = &gep_op.get_attr_gep_indices(self.ctx).unwrap().0;
        for idx_attr in indices {
            write!(output, ", ").unwrap();
            match idx_attr {
                GepIndexAttr::Constant(val) => {
                    write!(output, "i32 {val}").unwrap();
                }
                GepIndexAttr::OperandIdx(operand_idx) => {
                    let val = op_ref.get_operand(*operand_idx);
                    self.export_type(val.get_type(self.ctx), output)?;
                    write!(output, " ").unwrap();
                    self.export_value(val, value_names, output)?;
                }
            }
        }
        writeln!(output).unwrap();
        Ok(())
    }
}
