/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Statement translation: MIR statements → `dialect-mir` operations.
//!
//! Handles MIR statements like assignments, storage markers, and projections.
//!
//! # Supported Statements
//!
//! | Statement Kind      | Translation                                          |
//! |---------------------|------------------------------------------------------|
//! | `Assign(_l, rv)`    | Rvalue → ops; result stored into `_l`'s alloca slot  |
//! | `*ptr = val`        | `mir.store`                                          |
//! | `s.field = val`     | `mir.field_addr` + `mir.store` through the slot      |
//! | `StorageLive`       | `mir.storage_live` (lifetime marker)                 |
//! | `StorageDead`       | `mir.storage_dead` (lifetime marker)                 |
//! | `Nop`               | Skipped                                              |
//!
//! # Projections
//!
//! Handles up to 2-level projections:
//! - `*ptr` → Store through pointer
//! - `s.field` → Field-address from the slot, then `mir.store`
//! - `(*ptr).field` → Load pointer, compute field address, store
//! - `s.outer.inner` → Chained field-address from the slot, then store
//! - `(*slice)[k] = v` → Load slice fat ptr, extract data ptr, GEP offset, store

use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::rvalue;
use crate::translator::values::ValueMap;
use dialect_mir::ops::{MirStorageDeadOp, MirStorageLiveOp, MirStoreOp};
use pliron::basic_block::BasicBlock;
use pliron::builtin::types::{IntegerType, Signedness};
use pliron::context::{Context, Ptr};
use pliron::input_err;
use pliron::location::{Located, Location};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::printable::Printable;
use pliron::r#type::Typed;
use pliron::utils::apint::APInt;
use pliron::value::Value;
use rustc_public::mir;
use std::num::NonZeroUsize;

/// Translates a MIR statement to one or more `dialect-mir` operations.
///
/// # Returns
///
/// The last inserted operation (for chaining), or `prev_op` if no ops were created.
/// For `Rvalue::Use`, no operation is created - just updates `value_map`.
pub fn translate_statement(
    ctx: &mut Context,
    body: &mir::Body,
    stmt: &mir::Statement,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) -> TranslationResult<Option<Ptr<Operation>>> {
    // Use Debug representation of the span as location
    let loc = Location::Named {
        name: format!("{:?}", stmt.span),
        child_loc: Box::new(Location::Unknown),
    };

    match &stmt.kind {
        mir::StatementKind::Assign(place, rvalue) => {
            // Translate the Rvalue to get the value being assigned
            let (rvalue_op_opt, result_value, last_inserted) = rvalue::translate_rvalue(
                ctx,
                body,
                rvalue,
                value_map,
                block_ptr,
                prev_op,
                loc.clone(),
            )?;

            // Map the result to the place (local variable)
            if place.projection.is_empty() {
                // Simple local assignment: write the rvalue into the local's
                // stack slot (`mir.store local_slot, value`). ZST locals
                // (no slot) are silently skipped -- nothing to materialise.
                let local = place.local;

                // Insert the rvalue operation if it's not None
                // For Rvalue::Use, rvalue_op_opt is None (no operation to insert)
                // For other Rvalues (like CheckedAdd), we need to insert the operation
                let current_prev = if let Some(rvalue_op) = rvalue_op_opt {
                    if let Some(prev) = last_inserted {
                        rvalue_op.insert_after(ctx, prev);
                    } else if let Some(prev) = prev_op {
                        rvalue_op.insert_after(ctx, prev);
                    } else {
                        rvalue_op.insert_at_front(block_ptr, ctx);
                    }
                    Some(rvalue_op)
                } else {
                    // For Rvalue::Use, return the last inserted operation (field extraction if any)
                    // If last_inserted is None, we return prev_op
                    last_inserted.or(prev_op)
                };

                let store_op =
                    value_map.store_local(ctx, local, result_value, block_ptr, current_prev);
                Ok(store_op.or(current_prev))
            } else if place.projection.len() == 1 {
                match &place.projection[0] {
                    mir::ProjectionElem::Deref => {
                        // *ptr = value (Store)
                        // Translate the pointer (base)
                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };

                        // Determine current_prev based on rvalue insertion
                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        // Re-translate place with updated prev_op to ensure ordering
                        let (ptr_val, prev_op_after_ptr) = rvalue::translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;

                        // Create Store Op
                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],                      // No results
                            vec![ptr_val, result_value], // ptr, value
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);

                        if let Some(prev) = prev_op_after_ptr {
                            store_op.insert_after(ctx, prev);
                        } else {
                            // This implies block was empty and both rvalue and place didn't insert ops?
                            // Or they inserted at front.
                            store_op.insert_at_front(block_ptr, ctx);
                        }

                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::Field(field_idx, field_ty) => {
                        // struct.field = value (field assignment)
                        //
                        // Alloca model: compute the field's address from the
                        // local's slot via [`MirFieldAddrOp`] and store
                        // directly. This keeps the write addressable by
                        // `mem2reg` and avoids rebuilding the whole aggregate
                        // on every field update.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let Some(slot) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for field assignment",
                                    local
                                ))
                            );
                        };

                        let field_type = types::translate_type(ctx, field_ty)?;
                        let slot_mutable = pointer_is_mutable(ctx, slot);
                        let field_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            field_type,
                            slot_mutable,
                            pointer_address_space(ctx, slot),
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let field_addr_op = Operation::new(
                            ctx,
                            MirFieldAddrOp::get_concrete_op_info(),
                            vec![field_ptr_ty],
                            vec![slot],
                            vec![],
                            0,
                        );
                        field_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        MirFieldAddrOp::new(field_addr_op).set_attr_field_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                        );
                        if let Some(prev) = current_prev {
                            field_addr_op.insert_after(ctx, prev);
                        } else {
                            field_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let field_ptr = field_addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![field_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, field_addr_op);
                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::ConstantIndex {
                        offset,
                        min_length: _,
                        from_end,
                    } => {
                        // arr[const_idx] = value.
                        //
                        // Alloca model: locate the element via
                        // `MirConstantOp` + `MirArrayElementAddrOp` from the
                        // local's slot and emit `mir.store`.

                        if *from_end {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "ConstantIndex with from_end=true not yet supported for writes"
                                )
                            );
                        }

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let index = *offset as usize;
                        let Some(arr_ptr) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for array element assignment",
                                    local
                                ))
                            );
                        };

                        let (element_ty, address_space) =
                            slot_array_element_ty(ctx, arr_ptr, &loc)?;

                        use dialect_mir::ops::MirConstantOp;
                        use pliron::builtin::attributes::IntegerAttr;

                        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed);
                        let index_apint =
                            APInt::from_i64(index as i64, NonZeroUsize::new(64).unwrap());
                        let index_attr = IntegerAttr::new(i64_ty, index_apint);

                        let const_op_ptr = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![i64_ty.into()],
                            vec![],
                            vec![],
                            0,
                        );
                        const_op_ptr.deref_mut(ctx).set_loc(loc.clone());
                        MirConstantOp::new(const_op_ptr).set_attr_value(ctx, index_attr);

                        if let Some(prev) = current_prev {
                            const_op_ptr.insert_after(ctx, prev);
                        } else {
                            const_op_ptr.insert_at_front(block_ptr, ctx);
                        }
                        current_prev = Some(const_op_ptr);
                        let index_value = const_op_ptr.deref(ctx).get_result(0);

                        let store_op = emit_array_element_store(
                            ctx,
                            arr_ptr,
                            index_value,
                            result_value,
                            element_ty,
                            address_space,
                            block_ptr,
                            current_prev,
                            loc,
                        );
                        Ok(Some(store_op))
                    }
                    mir::ProjectionElem::Index(index_local) => {
                        // arr[i] = value with runtime index.
                        //
                        // Alloca model: fetch the index (via `load_local`
                        // through translate_place), GEP from the array's
                        // slot, and `mir.store` the value.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let local = place.local;
                        let Some(arr_ptr) = value_map.get_slot(local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for runtime index write",
                                    local
                                ))
                            );
                        };

                        let index_place = mir::Place {
                            local: *index_local,
                            projection: vec![],
                        };
                        let (index_value, prev_op_after_index) = rvalue::translate_place(
                            ctx,
                            body,
                            &index_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;
                        current_prev = prev_op_after_index;

                        let (element_ty, address_space) =
                            slot_array_element_ty(ctx, arr_ptr, &loc)?;

                        let store_op = emit_array_element_store(
                            ctx,
                            arr_ptr,
                            index_value,
                            result_value,
                            element_ty,
                            address_space,
                            block_ptr,
                            current_prev,
                            loc,
                        );
                        Ok(Some(store_op))
                    }
                    _ => input_err!(
                        loc,
                        TranslationErr::unsupported(
                            "Assignments to projections other than Deref, Field, ConstantIndex, and Index not yet implemented"
                        )
                    ),
                }
            } else if place.projection.len() == 2 {
                // Handle 2-level projections
                match (&place.projection[0], &place.projection[1]) {
                    (
                        mir::ProjectionElem::Deref,
                        mir::ProjectionElem::Field(field_idx, field_ty),
                    ) => {
                        // `(*ptr).field = value`.
                        //
                        // Alloca model: compute the field's address with
                        // `MirFieldAddrOp` applied to the pointer directly
                        // and store the new value with `MirStoreOp`.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let base_place = mir::Place {
                            local: place.local,
                            projection: vec![],
                        };
                        let (ptr_val, prev_op_after_ptr) = rvalue::translate_place(
                            ctx,
                            body,
                            &base_place,
                            value_map,
                            block_ptr,
                            current_prev,
                            loc.clone(),
                        )?;
                        current_prev = prev_op_after_ptr.or(current_prev);

                        let ptr_mutable = pointer_is_mutable(ctx, ptr_val);
                        let ptr_addr_space = pointer_address_space(ctx, ptr_val);

                        let field_type = types::translate_type(ctx, field_ty)?;
                        let field_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            field_type,
                            ptr_mutable,
                            ptr_addr_space,
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let addr_op = Operation::new(
                            ctx,
                            MirFieldAddrOp::get_concrete_op_info(),
                            vec![field_ptr_ty],
                            vec![ptr_val],
                            vec![],
                            0,
                        );
                        addr_op.deref_mut(ctx).set_loc(loc.clone());
                        MirFieldAddrOp::new(addr_op).set_attr_field_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(*field_idx as u32),
                        );
                        if let Some(prev) = current_prev {
                            addr_op.insert_after(ctx, prev);
                        } else {
                            addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let field_ptr = addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![field_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, addr_op);

                        Ok(Some(store_op))
                    }
                    (
                        mir::ProjectionElem::Field(outer_field_idx, outer_field_ty),
                        mir::ProjectionElem::Field(inner_field_idx, inner_field_ty),
                    ) => {
                        // `_local.outer.inner = value`.
                        //
                        // Alloca model: compose two `MirFieldAddrOp`s from the
                        // local's slot to reach the inner field's address,
                        // then store directly. `mem2reg` folds the chained
                        // addresses back into scalar field updates.

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let Some(slot) = value_map.get_slot(place.local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {} has no alloca slot for nested field assignment",
                                    Into::<usize>::into(place.local)
                                ))
                            );
                        };
                        let slot_mutable = pointer_is_mutable(ctx, slot);
                        let slot_addr_space = pointer_address_space(ctx, slot);

                        let outer_field_type = types::translate_type(ctx, outer_field_ty)?;
                        let outer_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            outer_field_type,
                            slot_mutable,
                            slot_addr_space,
                        )
                        .into();

                        use dialect_mir::ops::MirFieldAddrOp;
                        let outer_addr_op = Operation::new(
                            ctx,
                            MirFieldAddrOp::get_concrete_op_info(),
                            vec![outer_ptr_ty],
                            vec![slot],
                            vec![],
                            0,
                        );
                        outer_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        MirFieldAddrOp::new(outer_addr_op).set_attr_field_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(*outer_field_idx as u32),
                        );
                        if let Some(prev) = current_prev {
                            outer_addr_op.insert_after(ctx, prev);
                        } else {
                            outer_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        current_prev = Some(outer_addr_op);
                        let outer_ptr = outer_addr_op.deref(ctx).get_result(0);

                        let inner_field_type = types::translate_type(ctx, inner_field_ty)?;
                        let inner_ptr_ty = dialect_mir::types::MirPtrType::get(
                            ctx,
                            inner_field_type,
                            slot_mutable,
                            slot_addr_space,
                        )
                        .into();
                        let inner_addr_op = Operation::new(
                            ctx,
                            MirFieldAddrOp::get_concrete_op_info(),
                            vec![inner_ptr_ty],
                            vec![outer_ptr],
                            vec![],
                            0,
                        );
                        inner_addr_op.deref_mut(ctx).set_loc(loc.clone());
                        MirFieldAddrOp::new(inner_addr_op).set_attr_field_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(*inner_field_idx as u32),
                        );
                        if let Some(prev) = current_prev {
                            inner_addr_op.insert_after(ctx, prev);
                        } else {
                            inner_addr_op.insert_at_front(block_ptr, ctx);
                        }
                        let inner_ptr = inner_addr_op.deref(ctx).get_result(0);

                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![inner_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, inner_addr_op);

                        Ok(Some(store_op))
                    }
                    (
                        mir::ProjectionElem::Deref,
                        mir::ProjectionElem::ConstantIndex {
                            offset,
                            min_length: _,
                            from_end,
                        },
                    ) => {
                        // `(*slice_local)[const_idx] = value`, where the
                        // local is a `&mut [T]` (or `*mut [T]`). The slot
                        // holds the slice fat pointer; we load it, take
                        // the data-pointer half, GEP to the constant
                        // offset, and store.
                        if *from_end {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(
                                    "Deref->ConstantIndex with from_end=true not yet supported for writes"
                                )
                            );
                        }

                        let mut current_prev = prev_op;
                        if let Some(rvalue_op) = rvalue_op_opt {
                            if let Some(prev) = last_inserted {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else if let Some(prev) = prev_op {
                                rvalue_op.insert_after(ctx, prev);
                                current_prev = Some(rvalue_op);
                            } else {
                                rvalue_op.insert_at_front(block_ptr, ctx);
                                current_prev = Some(rvalue_op);
                            }
                        } else if let Some(prev) = last_inserted {
                            current_prev = Some(prev);
                        }

                        let Some(slot) = value_map.get_slot(place.local) else {
                            return input_err!(
                                loc,
                                TranslationErr::unsupported(format!(
                                    "Local {:?} has no alloca slot for Deref->ConstantIndex write",
                                    place.local
                                ))
                            );
                        };

                        // The slot's pointee must be a slice. Anything else
                        // (e.g. a thin `*mut T` would only be a single Deref)
                        // is a structural mismatch.
                        let element_ty = {
                            let slot_ty = slot.get_type(ctx);
                            let slot_ty_ref = slot_ty.deref(ctx);
                            let slot_ptr =
                                slot_ty_ref.downcast_ref::<dialect_mir::types::MirPtrType>();
                            let slice_ty = slot_ptr.and_then(|p| {
                                p.pointee
                                    .deref(ctx)
                                    .downcast_ref::<dialect_mir::types::MirSliceType>()
                                    .map(|s| s.element_type())
                            });
                            match slice_ty {
                                Some(elem) => elem,
                                None => {
                                    let ty_dbg = format!("{}", slot_ty.disp(ctx));
                                    return input_err!(
                                        loc,
                                        TranslationErr::unsupported(format!(
                                            "Deref->ConstantIndex write expects slot of MirPtrType<MirSliceType<T>>, got {}",
                                            ty_dbg
                                        ))
                                    );
                                }
                            }
                        };

                        // Load slot -> MirSliceType<T> value.
                        let slice_ty: Ptr<pliron::r#type::TypeObj> =
                            dialect_mir::types::MirSliceType::get(ctx, element_ty).into();
                        use dialect_mir::ops::MirLoadOp;
                        let load_op = Operation::new(
                            ctx,
                            MirLoadOp::get_concrete_op_info(),
                            vec![slice_ty],
                            vec![slot],
                            vec![],
                            0,
                        );
                        load_op.deref_mut(ctx).set_loc(loc.clone());
                        if let Some(prev) = current_prev {
                            load_op.insert_after(ctx, prev);
                        } else {
                            load_op.insert_at_front(block_ptr, ctx);
                        }
                        let slice_val = Value::OpResult {
                            op: load_op,
                            res_idx: 0,
                        };

                        // Extract data pointer (field 0 of the fat pointer).
                        // Mutable + generic addrspace matches how the
                        // intrinsic and rvalue paths model slice data ptrs.
                        let data_ptr_ty: Ptr<pliron::r#type::TypeObj> =
                            dialect_mir::types::MirPtrType::get_generic(ctx, element_ty, true)
                                .into();
                        use dialect_mir::ops::MirExtractFieldOp;
                        let extract_op = Operation::new(
                            ctx,
                            MirExtractFieldOp::get_concrete_op_info(),
                            vec![data_ptr_ty],
                            vec![slice_val],
                            vec![],
                            0,
                        );
                        extract_op.deref_mut(ctx).set_loc(loc.clone());
                        MirExtractFieldOp::new(extract_op).set_attr_index(
                            ctx,
                            dialect_mir::attributes::FieldIndexAttr(0),
                        );
                        extract_op.insert_after(ctx, load_op);
                        let data_ptr = Value::OpResult {
                            op: extract_op,
                            res_idx: 0,
                        };

                        // Constant offset (i64 signless).
                        use dialect_mir::ops::MirConstantOp;
                        use pliron::builtin::attributes::IntegerAttr;
                        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
                        let idx_apint =
                            APInt::from_i64(*offset as i64, NonZeroUsize::new(64).unwrap());
                        let idx_attr = IntegerAttr::new(i64_ty, idx_apint);
                        let const_op = Operation::new(
                            ctx,
                            MirConstantOp::get_concrete_op_info(),
                            vec![i64_ty.into()],
                            vec![],
                            vec![],
                            0,
                        );
                        const_op.deref_mut(ctx).set_loc(loc.clone());
                        MirConstantOp::new(const_op).set_attr_value(ctx, idx_attr);
                        const_op.insert_after(ctx, extract_op);
                        let idx_val = Value::OpResult {
                            op: const_op,
                            res_idx: 0,
                        };

                        // GEP: data_ptr + offset -> element pointer.
                        use dialect_mir::ops::MirPtrOffsetOp;
                        let offset_op = Operation::new(
                            ctx,
                            MirPtrOffsetOp::get_concrete_op_info(),
                            vec![data_ptr_ty],
                            vec![data_ptr, idx_val],
                            vec![],
                            0,
                        );
                        offset_op.deref_mut(ctx).set_loc(loc.clone());
                        offset_op.insert_after(ctx, const_op);
                        let elem_ptr = Value::OpResult {
                            op: offset_op,
                            res_idx: 0,
                        };

                        // Store value through element pointer.
                        let store_op = Operation::new(
                            ctx,
                            MirStoreOp::get_concrete_op_info(),
                            vec![],
                            vec![elem_ptr, result_value],
                            vec![],
                            0,
                        );
                        store_op.deref_mut(ctx).set_loc(loc);
                        store_op.insert_after(ctx, offset_op);

                        Ok(Some(store_op))
                    }
                    _ => input_err!(
                        loc,
                        TranslationErr::unsupported(format!(
                            "2-level projection {:?} -> {:?} not yet implemented for assignment",
                            place.projection[0], place.projection[1]
                        ))
                    ),
                }
            } else {
                input_err!(
                    loc,
                    TranslationErr::unsupported(format!(
                        "Complex places ({} projections) not yet implemented",
                        place.projection.len()
                    ))
                )
            }
        }
        mir::StatementKind::StorageLive(_local) => {
            // StorageLive marker
            let op = Operation::new(
                ctx,
                MirStorageLiveOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(Some(op))
        }
        mir::StatementKind::StorageDead(_local) => {
            // StorageDead marker
            let op = Operation::new(
                ctx,
                MirStorageDeadOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            op.deref_mut(ctx).set_loc(loc);

            if let Some(prev) = prev_op {
                op.insert_after(ctx, prev);
            } else {
                op.insert_at_front(block_ptr, ctx);
            }
            Ok(Some(op))
        }
        mir::StatementKind::Nop => {
            // No-op statement, skip
            Ok(prev_op)
        }

        // Codegen-irrelevant statements: borrow-check / type-system / coverage
        // hints that have no runtime effect. Skipping is correct.
        mir::StatementKind::FakeRead(..)
        | mir::StatementKind::Retag(..)
        | mir::StatementKind::PlaceMention(..)
        | mir::StatementKind::AscribeUserType { .. }
        | mir::StatementKind::Coverage(..)
        | mir::StatementKind::ConstEvalCounter => Ok(prev_op),

        // `Assume` is an optimisation hint with no observable effect; safe to skip.
        mir::StatementKind::Intrinsic(mir::NonDivergingIntrinsic::Assume(_)) => Ok(prev_op),

        // `Intrinsic(CopyNonOverlapping)` is the user-visible memcpy emitted by
        // `core::ptr::copy_nonoverlapping` (and its `<[T]>::copy_from_slice` /
        // `ptr::write_bytes` wrappers). It surfaces as a *statement* with
        // `(src, dst, count)` operands — not a `Terminator::Call` — so we
        // reshape it into a void `mir.call` carrying the
        // `CALLEE_COPY_NONOVERLAPPING` placeholder. mir-lower replaces the
        // placeholder with `@llvm.memcpy.p0.p0.i64` and recovers `sizeof(T)`
        // from the `dst` operand's `MirPtrType`.
        mir::StatementKind::Intrinsic(mir::NonDivergingIntrinsic::CopyNonOverlapping(copy)) => {
            translate_copy_nonoverlapping(ctx, body, copy, value_map, block_ptr, prev_op, loc)
        }
        mir::StatementKind::SetDiscriminant { .. } => input_err!(
            loc,
            TranslationErr::unsupported(
                "SetDiscriminant statements are not yet supported on the device; \
                 until they are lowered, enum discriminant writes would be silently dropped",
            )
        ),
    }
}

/// Lower `NonDivergingIntrinsic::CopyNonOverlapping` into a void `mir.call`
/// carrying the [`CALLEE_COPY_NONOVERLAPPING`] placeholder name.
///
/// The MIR statement carries three operands `(src, dst, count)`. We translate
/// each operand, then construct a `mir.call` with zero results. mir-lower
/// recognizes the placeholder and replaces the call with
/// `@llvm.memcpy.p0.p0.i64(dst, src, count * sizeof(T), false)` —
/// `sizeof(T)` recovered from the `dst` operand's `MirPtrType`.
///
/// [`CALLEE_COPY_NONOVERLAPPING`]: dialect_mir::rust_intrinsics::CALLEE_COPY_NONOVERLAPPING
fn translate_copy_nonoverlapping(
    ctx: &mut Context,
    body: &mir::Body,
    copy: &mir::CopyNonOverlapping,
    value_map: &mut ValueMap,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> TranslationResult<Option<Ptr<Operation>>> {
    use dialect_mir::ops::MirCallOp;
    use dialect_mir::rust_intrinsics;
    use pliron::builtin::attributes::StringAttr;

    let mut last_op = prev_op;

    let (src_val, prev_after_src) = rvalue::translate_operand(
        ctx,
        body,
        &copy.src,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = prev_after_src;

    let (dst_val, prev_after_dst) = rvalue::translate_operand(
        ctx,
        body,
        &copy.dst,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = prev_after_dst;

    let (count_val, prev_after_count) = rvalue::translate_operand(
        ctx,
        body,
        &copy.count,
        value_map,
        block_ptr,
        last_op,
        loc.clone(),
    )?;
    last_op = prev_after_count;

    let call_op = Operation::new(
        ctx,
        MirCallOp::get_concrete_op_info(),
        vec![], // void: zero results
        vec![src_val, dst_val, count_val],
        vec![],
        0,
    );
    call_op.deref_mut(ctx).set_loc(loc);

    let callee_attr = StringAttr::new(rust_intrinsics::CALLEE_COPY_NONOVERLAPPING.into());
    call_op.deref_mut(ctx).attributes.0.insert(
        pliron::identifier::Identifier::try_from("callee").unwrap(),
        callee_attr.into(),
    );

    if let Some(prev) = last_op {
        call_op.insert_after(ctx, prev);
    } else {
        call_op.insert_at_front(block_ptr, ctx);
    }

    Ok(Some(call_op))
}

/// Extract the element type and address space from a pointer that points
/// to an array.
///
/// Used by the statement-level element write helpers. Returns a structured
/// error when the pointer's pointee isn't a [`MirArrayType`], which signals
/// a structural mismatch (most likely the wrong MIR projection reaching
/// this path).
fn slot_array_element_ty(
    ctx: &pliron::context::Context,
    arr_ptr: Value,
    loc: &Location,
) -> TranslationResult<(pliron::context::Ptr<pliron::r#type::TypeObj>, u32)> {
    let arr_ptr_ty = arr_ptr.get_type(ctx);
    let arr_ptr_ty_ref = arr_ptr_ty.deref(ctx);
    let mir_ptr_ty = arr_ptr_ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::unsupported("Array-index slot is not a MirPtrType")
            )
        })?;
    let address_space = mir_ptr_ty.address_space;
    let pointee_ref = mir_ptr_ty.pointee.deref(ctx);
    let element_ty = pointee_ref
        .downcast_ref::<dialect_mir::types::MirArrayType>()
        .ok_or_else(|| {
            pliron::input_error!(
                loc.clone(),
                TranslationErr::unsupported("Array-index slot pointee is not MirArrayType",)
            )
        })?
        .element_type();
    Ok((element_ty, address_space))
}

/// Emit `mir.array_element_addr` + `mir.store` to assign `value` into
/// `array_ptr[index]`, returning the `mir.store` op.
///
/// The caller owns positioning (`prev_op`): we chain the address op after
/// it, then chain the store after the address op.
#[allow(clippy::too_many_arguments)]
fn emit_array_element_store(
    ctx: &mut pliron::context::Context,
    array_ptr: Value,
    index: Value,
    value: Value,
    element_ty: pliron::context::Ptr<pliron::r#type::TypeObj>,
    address_space: u32,
    block_ptr: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    loc: Location,
) -> Ptr<Operation> {
    let elem_ptr_ty =
        dialect_mir::types::MirPtrType::get(ctx, element_ty, true, address_space).into();

    use dialect_mir::ops::MirArrayElementAddrOp;
    let addr_op = Operation::new(
        ctx,
        MirArrayElementAddrOp::get_concrete_op_info(),
        vec![elem_ptr_ty],
        vec![array_ptr, index],
        vec![],
        0,
    );
    addr_op.deref_mut(ctx).set_loc(loc.clone());
    match prev_op {
        Some(prev) => addr_op.insert_after(ctx, prev),
        None => addr_op.insert_at_front(block_ptr, ctx),
    }
    let elem_ptr = addr_op.deref(ctx).get_result(0);

    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![elem_ptr, value],
        vec![],
        0,
    );
    store_op.deref_mut(ctx).set_loc(loc);
    store_op.insert_after(ctx, addr_op);
    store_op
}

/// Return `true` if the pointer value's type is a mutable [`MirPtrType`].
///
/// Slots emitted by the entry-block alloca loop are always mutable, but
/// callers of the statement module sometimes thread pointers coming from
/// other sources (loads, field-addr ops, ...), which may be immutable.
/// Derived addresses inherit the base pointer's mutability to keep pliron
/// type checking consistent.
fn pointer_is_mutable(ctx: &pliron::context::Context, ptr: Value) -> bool {
    let ty = ptr.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .is_some_and(|p| p.is_mutable)
}

/// Return the address space of a pointer value. Defaults to 0 (the generic
/// address space) if the value is not a [`MirPtrType`].
fn pointer_address_space(ctx: &pliron::context::Context, ptr: Value) -> u32 {
    let ty = ptr.get_type(ctx);
    let ty_ref = ty.deref(ctx);
    ty_ref
        .downcast_ref::<dialect_mir::types::MirPtrType>()
        .map(|p| p.address_space)
        .unwrap_or(0)
}
