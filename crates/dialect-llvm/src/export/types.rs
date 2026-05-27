/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM type printing.

use std::fmt::Write;

use pliron::{
    builtin::types::{FP32Type, FP64Type, IntegerType},
    context::Ptr,
    r#type::TypeObj,
};

use crate::types::{HalfType, PointerType, StructType, VoidType};

use super::state::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_type(&self, ty: Ptr<TypeObj>, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if addrspace != 0 {
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
