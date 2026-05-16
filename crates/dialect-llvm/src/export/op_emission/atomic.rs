/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use std::collections::HashMap;
use std::fmt::Write;

use pliron::{operation::Operation, r#type::Typed, value::Value};

use crate::{ops, ops::atomic::LlvmAtomicOpInterface, types::PointerType};

use super::super::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_atomic_load_op(
        &self,
        op_ref: &Operation,
        atomic_load: &ops::AtomicLoadOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let res_name = value_names.get(&res).unwrap();
        let ty = res.get_type(self.ctx);
        let syncscope_str = ops::atomic::format_syncscope(&atomic_load.syncscope(self.ctx));
        let ordering_str = ops::atomic::format_ordering(&atomic_load.ordering(self.ctx));

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        write!(output, "  {res_name} = load atomic ").unwrap();
        self.export_type(ty, output)?;
        if addrspace != 0 {
            write!(output, ", ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, ", ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(ty);
        writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
        Ok(())
    }

    pub(super) fn export_atomic_store_op(
        &self,
        op_ref: &Operation,
        atomic_store: &ops::AtomicStoreOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let val = op_ref.get_operand(0);
        let ptr = op_ref.get_operand(1);
        let syncscope_str = ops::atomic::format_syncscope(&atomic_store.syncscope(self.ctx));
        let ordering_str = ops::atomic::format_ordering(&atomic_store.ordering(self.ctx));

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        write!(output, "  store atomic ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        if addrspace != 0 {
            write!(output, ", ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, ", ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        let align = self.natural_alignment(val.get_type(self.ctx));
        writeln!(output, "{syncscope_str} {ordering_str}, align {align}").unwrap();
        Ok(())
    }

    pub(super) fn export_atomic_rmw_op(
        &self,
        op_ref: &Operation,
        atomic_rmw: &ops::AtomicRmwOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let val = op_ref.get_operand(1);
        let res_name = value_names.get(&res).unwrap();
        let rmw_kind_str = ops::atomic::format_rmw_kind(&atomic_rmw.rmw_kind(self.ctx));
        let syncscope_str = ops::atomic::format_syncscope(&atomic_rmw.syncscope(self.ctx));
        let ordering_str = ops::atomic::format_ordering(&atomic_rmw.ordering(self.ctx));

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        write!(output, "  {res_name} = atomicrmw {rmw_kind_str} ").unwrap();
        if addrspace != 0 {
            write!(output, "ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, "ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val.get_type(self.ctx), output)?;
        write!(output, " ").unwrap();
        self.export_value(val, value_names, output)?;
        writeln!(output, "{syncscope_str} {ordering_str}").unwrap();
        Ok(())
    }

    pub(super) fn export_atomic_cmpxchg_op(
        &self,
        op_ref: &Operation,
        atomic_cmpxchg: &ops::AtomicCmpxchgOp,
        value_names: &HashMap<Value, String>,
        output: &mut String,
    ) -> Result<(), String> {
        let res = op_ref.get_result(0);
        let ptr = op_ref.get_operand(0);
        let cmp = op_ref.get_operand(1);
        let new_val = op_ref.get_operand(2);
        let res_name = value_names.get(&res).unwrap();
        let success_ord_str =
            ops::atomic::format_ordering(&atomic_cmpxchg.success_ordering(self.ctx));
        let failure_ord_str =
            ops::atomic::format_ordering(&atomic_cmpxchg.failure_ordering(self.ctx));
        let syncscope_str = ops::atomic::format_syncscope(&atomic_cmpxchg.syncscope(self.ctx));
        let val_ty = cmp.get_type(self.ctx);

        let addrspace = ptr
            .get_type(self.ctx)
            .deref(self.ctx)
            .downcast_ref::<PointerType>()
            .map_or(0, PointerType::address_space);

        let struct_name = format!("{res_name}.cx");
        write!(output, "  {struct_name} = cmpxchg ").unwrap();
        if addrspace != 0 {
            write!(output, "ptr addrspace({addrspace}) ").unwrap();
        } else {
            write!(output, "ptr ").unwrap();
        }
        self.export_value(ptr, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(cmp, value_names, output)?;
        write!(output, ", ").unwrap();
        self.export_type(val_ty, output)?;
        write!(output, " ").unwrap();
        self.export_value(new_val, value_names, output)?;
        writeln!(
            output,
            "{syncscope_str} {success_ord_str} {failure_ord_str}"
        )
        .unwrap();

        write!(output, "  {res_name} = extractvalue {{ ").unwrap();
        self.export_type(val_ty, output)?;
        writeln!(output, ", i1 }} {struct_name}, 0").unwrap();
        Ok(())
    }

    pub(super) fn export_fence_op(&self, fence: &ops::FenceOp, output: &mut String) {
        let syncscope_str = ops::atomic::format_syncscope(&fence.syncscope(self.ctx));
        let ordering_str = ops::atomic::format_ordering(&fence.ordering(self.ctx));
        writeln!(output, "  fence{syncscope_str} {ordering_str}").unwrap();
    }
}
