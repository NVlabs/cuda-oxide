/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM type printing.

use std::fmt::Write;

use pliron::{
    builtin::types::{FP32Type, FP64Type, IntegerType},
    context::Ptr,
    operation::Operation,
    r#type::{TypeObj, Typed},
    value::Value,
};

use crate::{
    ops,
    types::{HalfType, PointerType, StructType, VoidType},
};

use super::{config::NvvmIrDialect, state::ModuleExportState};

impl<'a> ModuleExportState<'a> {
    pub(super) fn type_to_string(&self, ty: Ptr<TypeObj>) -> Result<String, String> {
        let mut output = String::new();
        self.export_type(ty, &mut output)?;
        Ok(output)
    }

    pub(super) fn export_type(&self, ty: Ptr<TypeObj>, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if self.nvvm_ir_dialect == Some(NvvmIrDialect::TypedPointers) {
                if addrspace != 0 {
                    write!(output, "i8 addrspace({addrspace})*").unwrap();
                } else {
                    write!(output, "i8*").unwrap();
                }
            } else if addrspace != 0 {
                write!(output, "ptr addrspace({addrspace})").unwrap();
            } else {
                write!(output, "ptr").unwrap();
            }
        } else if ty_ref.is::<VoidType>() {
            write!(output, "void").unwrap();
        } else if ty_ref.is::<HalfType>() {
            write!(output, "half").unwrap();
        } else if ty_ref.is::<FP32Type>() {
            write!(output, "float").unwrap();
        } else if ty_ref.is::<FP64Type>() {
            write!(output, "double").unwrap();
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            write!(output, "{{ ").unwrap();
            for (i, elem_ty) in struct_ty.fields().enumerate() {
                if i > 0 {
                    write!(output, ", ").unwrap();
                }
                self.export_type(elem_ty, output)?;
            }
            write!(output, " }}").unwrap();
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            write!(output, "[{} x ", array_ty.size()).unwrap();
            self.export_type(array_ty.elem_type(), output)?;
            write!(output, "]").unwrap();
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            write!(output, "<{} x ", vec_ty.size()).unwrap();
            self.export_type(vec_ty.elem_type(), output)?;
            write!(output, ">").unwrap();
        } else {
            write!(output, "void /* unknown: {} */", ty_ref.disp(self.ctx)).unwrap();
        }
        Ok(())
    }

    pub(super) fn pointer_type_for_pointee(
        &self,
        pointee_ty: Ptr<TypeObj>,
        addrspace: u32,
    ) -> Result<String, String> {
        let mut ty = self.type_to_string(pointee_ty)?;
        if addrspace != 0 {
            ty.push_str(&format!(" addrspace({addrspace})*"));
        } else {
            ty.push('*');
        }
        Ok(ty)
    }

    pub(super) fn pointer_value_type(&self, val: Value) -> Result<String, String> {
        if let Some(ty) = self.pointer_value_type_from_defining_op(val)? {
            return Ok(ty);
        }

        if let Some(ty) = self.typed_pointer_value_types.get(&val) {
            return Ok(ty.clone());
        }

        self.type_to_string(val.get_type(self.ctx))
    }

    fn pointer_value_type_from_defining_op(&self, val: Value) -> Result<Option<String>, String> {
        let Some(defining_op) = val.defining_op() else {
            return Ok(None);
        };
        let op_ref = defining_op.deref(self.ctx);
        let op_obj = Operation::get_op_dyn(defining_op, self.ctx);
        let op_dyn = op_obj.as_ref();

        if let Some(alloca) = op_dyn.downcast_ref::<ops::AllocaOp>() {
            let elem_ty = alloca
                .get_attr_alloca_element_type(self.ctx)
                .ok_or("Missing alloca_element_type")?
                .get_type(self.ctx);
            return self.pointer_type_for_pointee(elem_ty, 0).map(Some);
        }

        if let Some(gep) = op_dyn.downcast_ref::<ops::GetElementPtrOp>() {
            let ptr = op_ref.get_operand(0);
            let elem_ty = gep
                .get_attr_gep_src_elem_type(self.ctx)
                .ok_or("Missing gep_src_elem_type")?
                .get_type(self.ctx);
            let addrspace = pointer_addrspace(ptr.get_type(self.ctx), self.ctx);
            let indices = gep.indices(self.ctx);
            let result_pointee =
                ops::GetElementPtrOp::indexed_type(self.ctx, elem_ty, &indices).unwrap_or(elem_ty);
            return self
                .pointer_type_for_pointee(result_pointee, addrspace)
                .map(Some);
        }

        Ok(None)
    }

    /// Compute natural alignment (in bytes) for a type.
    /// Used for atomic load/store which require explicit alignment in LLVM IR.
    pub(super) fn natural_alignment(&self, ty: Ptr<TypeObj>) -> u32 {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            let width = int_ty.width();
            // Alignment = ceil(width / 8), minimum 1
            std::cmp::max(1, width / 8)
        } else if ty_ref.is::<pliron::builtin::types::FP32Type>() {
            4
        } else if ty_ref.is::<pliron::builtin::types::FP64Type>() {
            8
        } else {
            // Default: 8 bytes (conservative for pointers, etc.)
            8
        }
    }
}

fn pointer_addrspace(ty: Ptr<TypeObj>, ctx: &pliron::context::Context) -> u32 {
    ty.deref(ctx)
        .downcast_ref::<PointerType>()
        .map_or(0, PointerType::address_space)
}
