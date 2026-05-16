/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{
    builtin::op_interfaces::{CallOpCallable, CallOpInterface},
    operation::Operation,
    r#type::Typed,
    value::Value,
};

use crate::{
    ops,
    types::{FuncType, VoidType},
};

use super::super::{ModuleExportState, strip_device_prefix};

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_call_op(
        &mut self,
        op_ref: &Operation,
        call: &ops::CallOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let callee = call.callee(self.ctx);

        let func_ty = call.callee_type(self.ctx);
        let func_ty_ref = func_ty.deref(self.ctx);
        let llvm_func_ty = func_ty_ref.downcast_ref::<FuncType>().unwrap();
        let ret_ty = llvm_func_ty.result_type();
        let is_void = ret_ty.deref(self.ctx).is::<VoidType>();

        if is_void {
            write!(output, "  call void").unwrap();
        } else {
            let res = op_ref.get_result(0);
            let res_name = value_names.get(&res).unwrap();
            write!(output, "  {res_name} = call ").unwrap();
            self.export_type(ret_ty, output)?;
        }

        let mut is_convergent_call = false;

        match callee {
            CallOpCallable::Direct(identifier) => {
                let name = identifier.to_string();
                let fixed_name = if name.starts_with("llvm_") {
                    name.replace('_', ".")
                } else {
                    strip_device_prefix(&name)
                };
                is_convergent_call = Self::is_convergent_intrinsic(&fixed_name);
                write!(output, " @{fixed_name}(").unwrap();
            }
            CallOpCallable::Indirect(val) => {
                write!(output, " ").unwrap();
                self.export_value(val, value_names, output).unwrap();
                write!(output, "(").unwrap();
            }
        }

        for (i, arg) in op_ref.operands().enumerate() {
            if i > 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(arg.get_type(self.ctx), output)?;
            write!(output, " ").unwrap();
            self.export_value(arg, value_names, output)?;
        }

        if is_convergent_call {
            writeln!(output, ") #0").unwrap();
            self.convergent_used = true;
        } else {
            writeln!(output, ")").unwrap();
        }

        Ok(())
    }
}
