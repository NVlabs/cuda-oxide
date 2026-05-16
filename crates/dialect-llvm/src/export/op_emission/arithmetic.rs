/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{operation::Operation, r#type::Typed, value::Value};

use crate::{attributes::ICmpPredicateAttr, ops};

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_fneg_op(
        &self,
        op_ref: &Operation,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let arg = op_ref.get_operand(0);

        write!(output, "  {res_name} = fneg ").unwrap();
        self.export_type(arg.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(arg, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_icmp_op(
        &self,
        op_ref: &Operation,
        icmp: &ops::ICmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred_str = match icmp.predicate(self.ctx) {
            ICmpPredicateAttr::EQ => "eq",
            ICmpPredicateAttr::NE => "ne",
            ICmpPredicateAttr::SLT => "slt",
            ICmpPredicateAttr::SLE => "sle",
            ICmpPredicateAttr::SGT => "sgt",
            ICmpPredicateAttr::SGE => "sge",
            ICmpPredicateAttr::ULT => "ult",
            ICmpPredicateAttr::ULE => "ule",
            ICmpPredicateAttr::UGT => "ugt",
            ICmpPredicateAttr::UGE => "uge",
        };

        write!(output, "  {res_name} = icmp {pred_str} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }

    pub(super) fn export_fcmp_op(
        &self,
        op_ref: &Operation,
        fcmp: &ops::FCmpOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let res_name = value_names.get(&res).unwrap();
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let pred_str = match fcmp.predicate(self.ctx) {
            crate::attributes::FCmpPredicateAttr::False => "false",
            crate::attributes::FCmpPredicateAttr::OEQ => "oeq",
            crate::attributes::FCmpPredicateAttr::OGT => "ogt",
            crate::attributes::FCmpPredicateAttr::OGE => "oge",
            crate::attributes::FCmpPredicateAttr::OLT => "olt",
            crate::attributes::FCmpPredicateAttr::OLE => "ole",
            crate::attributes::FCmpPredicateAttr::ONE => "one",
            crate::attributes::FCmpPredicateAttr::ORD => "ord",
            crate::attributes::FCmpPredicateAttr::UEQ => "ueq",
            crate::attributes::FCmpPredicateAttr::UGT => "ugt",
            crate::attributes::FCmpPredicateAttr::UGE => "uge",
            crate::attributes::FCmpPredicateAttr::ULT => "ult",
            crate::attributes::FCmpPredicateAttr::ULE => "ule",
            crate::attributes::FCmpPredicateAttr::UNE => "une",
            crate::attributes::FCmpPredicateAttr::UNO => "uno",
            crate::attributes::FCmpPredicateAttr::True => "true",
        };

        write!(output, "  {res_name} = fcmp {pred_str} ").unwrap();
        self.export_type(lhs.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(lhs, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_value(rhs, value_names, output)?;
        writeln!(output).unwrap();
        Ok(())
    }
}
