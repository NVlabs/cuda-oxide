/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM type printing.

use std::fmt::Write;

use pliron::{
    builtin::{
        type_interfaces::FunctionTypeInterface,
        types::{FP32Type, FP64Type, IntegerType},
    },
    r#type::TypeHandle,
};

use crate::types::{FuncType, HalfType, PointerType, StructType, VoidType};

use super::state::ModuleExportState;

impl<'a> ModuleExportState<'a> {
    pub(super) fn export_type(&self, ty: TypeHandle, output: &mut String) -> Result<(), String> {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            write!(output, "i{}", int_ty.width()).unwrap();
        } else if let Some(ptr_ty) = ty_ref.downcast_ref::<PointerType>() {
            let addrspace = ptr_ty.address_space();
            if self.legacy_typed_pointers() {
                self.export_canonical_pointer_type(addrspace, output);
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
            write!(output, "<{} x ", vec_ty.num_elements()).unwrap();
            self.export_type(vec_ty.elem_type(), output)?;
            write!(output, ">").unwrap();
        } else {
            return Err(format!(
                "cannot export unknown LLVM type `{}`",
                ty_ref.disp(self.ctx)
            ));
        }
        Ok(())
    }

    pub(super) fn is_pointer_type(&self, ty: TypeHandle) -> bool {
        ty.deref(self.ctx).is::<PointerType>()
    }

    pub(super) fn is_i8_type(&self, ty: TypeHandle) -> bool {
        ty.deref(self.ctx)
            .downcast_ref::<IntegerType>()
            .is_some_and(|integer| integer.width() == 8)
    }

    /// Print the canonical legacy representation of an erased pointer.
    pub(super) fn export_canonical_pointer_type(&self, addrspace: u32, output: &mut String) {
        write!(output, "i8").unwrap();
        if addrspace != 0 {
            write!(output, " addrspace({addrspace})").unwrap();
        }
        write!(output, "*").unwrap();
    }

    /// Print a typed pointer to `pointee` in `addrspace` for LLVM 7 syntax.
    pub(super) fn export_pointer_to(
        &self,
        pointee: TypeHandle,
        addrspace: u32,
        output: &mut String,
    ) -> Result<(), String> {
        self.export_type(pointee, output)?;
        if addrspace != 0 {
            write!(output, " addrspace({addrspace})").unwrap();
        }
        write!(output, "*").unwrap();
        Ok(())
    }

    /// Print an LLVM 7 function-pointer type using the same recursively
    /// canonicalized argument and result types as function declarations.
    pub(super) fn export_function_pointer_type(
        &self,
        function_type: TypeHandle,
        output: &mut String,
    ) -> Result<(), String> {
        let function_ref = function_type.deref(self.ctx);
        let function_type = function_ref.downcast_ref::<FuncType>().ok_or_else(|| {
            format!(
                "expected function type, got `{}`",
                function_ref.disp(self.ctx)
            )
        })?;

        self.export_type(function_type.result_type(), output)?;
        write!(output, " (").unwrap();
        for (index, argument) in function_type.arg_types().iter().enumerate() {
            if index != 0 {
                write!(output, ", ").unwrap();
            }
            self.export_type(*argument, output)?;
        }
        if function_type.is_var_arg() {
            if !function_type.arg_types().is_empty() {
                write!(output, ", ").unwrap();
            }
            write!(output, "...").unwrap();
        }
        write!(output, ")*").unwrap();
        Ok(())
    }

    /// Compute conservative ABI alignment (bytes) for a type.
    ///
    /// Used as the fallback when no explicit alignment is stamped on a
    /// load/store/alloca op. Required for atomic loads/stores (LLVM IR
    /// mandates explicit alignment) and for vectorization hints.
    pub(super) fn natural_alignment(&self, ty: TypeHandle) -> u32 {
        let ty_ref = ty.deref(self.ctx);
        if let Some(int_ty) = ty_ref.downcast_ref::<IntegerType>() {
            // ceil(width / 8), minimum 1.
            std::cmp::max(1, int_ty.width() / 8)
        } else if ty_ref.is::<FP32Type>() {
            4
        } else if ty_ref.is::<FP64Type>() {
            8
        } else if ty_ref.is::<HalfType>() {
            2
        } else if ty_ref.is::<PointerType>() {
            8
        } else if let Some(array_ty) = ty_ref.downcast_ref::<crate::types::ArrayType>() {
            // ABI alignment of `[N x T]` matches elem alignment.
            self.natural_alignment(array_ty.elem_type())
        } else if let Some(vec_ty) = ty_ref.downcast_ref::<crate::types::VectorType>() {
            // ABI alignment of an LLVM vector: power-of-2-rounded total width.
            let elem = self.natural_alignment(vec_ty.elem_type());
            let total = elem.saturating_mul(vec_ty.num_elements());
            let mut a = 1u32;
            while a.saturating_mul(2) <= total && a < 128 {
                a *= 2;
            }
            a
        } else if let Some(struct_ty) = ty_ref.downcast_ref::<StructType>() {
            // Max field alignment (1 if empty). May under-state a repr(align)
            // raise; the true alignment is carried on the op, not the type.
            struct_ty
                .fields()
                .map(|f| self.natural_alignment(f))
                .max()
                .unwrap_or(1)
        } else {
            // Conservative fallback for pointers and unknown types.
            8
        }
    }
}
