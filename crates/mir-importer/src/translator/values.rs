/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Value mapping: MIR locals → alloca slots and associated helper ops.
//!
//! Every non-ZST MIR local is backed by a single `mir.alloca` emitted at the
//! top of the function's entry block. Defs lower to `mir.store` and uses to
//! `mir.load`. This module owns the per-local slot map and the emitters for
//! the three slot operations.
//!
//! The `mem2reg` pass in `pipeline.rs` promotes the scalar slots back into
//! SSA before LLVM lowering, so at steady state the only `mir.alloca`s that
//! survive are those whose addresses actually escape.
//!
//! # Slot address-space inference
//!
//! Rust's reference / raw-pointer types carry no address-space information
//! (`&mut f32`, `*const u32` all translate to a generic `MirPtrType`). On
//! GPU, however, intermediate locals frequently end up holding pointers in
//! a concrete address space — e.g. `let p = &mut TILE_A[i]` on a
//! `SharedArray` produces an `addrspace(3)` pointer, yet the Rust local is
//! typed `&mut f32` (generic).
//!
//! Picking the slot's addrspace from Rust's declared type alone causes every
//! store of such a concrete-addrspace pointer to go through a
//! `mir.cast <PtrToPtr>` → LLVM `addrspacecast` → PTX `cvta.shared.u64`,
//! and subsequent loads of that pointer to hit the generic (runtime-
//! dispatched) store path instead of native `st.shared.*`.
//!
//! [`SlotAddrSpaceMap`] is a pre-scan over the MIR body that, per local,
//! infers the pointee address space from the *writes* into that local. The
//! answer is used by `body::emit_entry_allocas` via
//! [`align_pointer_addr_space`] to pick an alloca pointee that matches what
//! actually gets stored.

use dialect_mir::attributes::MirCastKindAttr;
use dialect_mir::ops::{MirAllocaOp, MirCastOp, MirLoadOp, MirStoreOp};
use dialect_mir::types::{MirPtrType, address_space};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::{TypeObj, Typed};
use pliron::value::Value;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::alloc::GlobalAlloc;
use rustc_public::ty::{ConstantKind, RigidTy, TyKind};

/// Maps MIR locals to their alloca slots.
///
/// # Invariants
///
/// - `slots.len() == num_locals` after construction.
/// - Each `Some(slot)` entry carries a value whose type is [`MirPtrType`];
///   the pointee is the local's Pliron-IR type.
/// - ZST locals (and the unit return slot) remain `None` in `slots`.
pub struct ValueMap {
    slots: Vec<Option<Value>>,
}

impl ValueMap {
    /// Creates a new map with capacity for the given number of MIR locals.
    pub fn new(num_locals: usize) -> Self {
        Self {
            slots: vec![None; num_locals],
        }
    }

    /// Return the alloca pointer backing `local`, or `None` if the local is
    /// ZST / has not been given a slot.
    pub fn get_slot(&self, local: mir::Local) -> Option<Value> {
        let idx: usize = local;
        self.slots.get(idx).copied().flatten()
    }

    /// Record the alloca pointer for `local`. Expected to be called once per
    /// non-ZST local during body setup in [`super::body::translate_body`].
    pub fn set_slot(&mut self, local: mir::Local, slot: Value) {
        let idx: usize = local;
        if idx < self.slots.len() {
            self.slots[idx] = Some(slot);
        }
    }

    /// Emit a `mir.alloca` for `elem_ty` and insert it into `block`.
    ///
    /// The result pointer lives in the generic address space and is marked
    /// mutable; the alloca's pointee carries the allocated element type. If
    /// `prev_op` is provided, the op is linked immediately after it; otherwise
    /// it is inserted at the front of `block`.
    ///
    /// Returns the inserted op and its result pointer value.
    pub fn emit_alloca(
        ctx: &mut Context,
        elem_ty: Ptr<TypeObj>,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> (Ptr<Operation>, Value) {
        let ptr_ty = MirPtrType::get_generic(ctx, elem_ty, /* is_mutable */ true);
        let op = Operation::new(
            ctx,
            MirAllocaOp::get_concrete_op_info(),
            vec![ptr_ty.into()],
            vec![],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        let result = op.deref(ctx).get_result(0);
        (op, result)
    }

    /// Emit `mir.load` from `local`'s slot. Returns `None` for ZST / unset
    /// locals.
    pub fn load_local(
        &self,
        ctx: &mut Context,
        local: mir::Local,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> Option<(Ptr<Operation>, Value)> {
        let slot = self.get_slot(local)?;
        let elem_ty = slot_pointee(ctx, slot);
        let op = Operation::new(
            ctx,
            MirLoadOp::get_concrete_op_info(),
            vec![elem_ty],
            vec![slot],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        let result = op.deref(ctx).get_result(0);
        Some((op, result))
    }

    /// Emit `mir.store` of `value` into `local`'s slot. Returns `None` for ZST
    /// / unset locals.
    ///
    /// When `value` is a pointer whose type differs from the slot's declared
    /// pointee (mutability, address space, or underlying pointee shape), a
    /// `mir.cast <PtrToPtr>` is inserted automatically. This bridges the
    /// common case where an rvalue produces a pointer in a concrete address
    /// space (e.g. `shared_alloc` returning `*mut T addrspace(3)`) while the
    /// local's Rust-declared type translates to a generic-addrspace pointer
    /// (e.g. `*mut SharedArray<T, N>` -> `*mut ()`). All pointers have the
    /// same runtime layout after lowering, so the cast is free.
    pub fn store_local(
        &self,
        ctx: &mut Context,
        local: mir::Local,
        value: Value,
        block: Ptr<BasicBlock>,
        prev_op: Option<Ptr<Operation>>,
    ) -> Option<Ptr<Operation>> {
        let slot = self.get_slot(local)?;
        let slot_elem_ty = slot_pointee(ctx, slot);
        let (value, prev_op) = maybe_ptr_coerce(ctx, value, slot_elem_ty, block, prev_op);
        let op = Operation::new(
            ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, value],
            vec![],
            0,
        );
        insert_at(ctx, op, block, prev_op);
        Some(op)
    }
}

/// If `value` is a pointer whose type differs from `target_ty` (also a
/// pointer), emit a `mir.cast <PtrToPtr>` that converts it. Returns the (new)
/// value and the (new) anchor op. Otherwise this is a no-op.
fn maybe_ptr_coerce(
    ctx: &mut Context,
    value: Value,
    target_ty: Ptr<TypeObj>,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) -> (Value, Option<Ptr<Operation>>) {
    let value_ty = value.get_type(ctx);
    if value_ty == target_ty {
        return (value, prev_op);
    }

    // Only auto-insert a PtrToPtr cast when both sides are already pointer
    // types; anything else is a genuine translation mismatch and should be
    // surfaced by the verifier, not papered over here.
    let value_is_ptr = value_ty.deref(ctx).downcast_ref::<MirPtrType>().is_some();
    let target_is_ptr = target_ty.deref(ctx).downcast_ref::<MirPtrType>().is_some();
    if !(value_is_ptr && target_is_ptr) {
        return (value, prev_op);
    }

    let cast_op = Operation::new(
        ctx,
        MirCastOp::get_concrete_op_info(),
        vec![target_ty],
        vec![value],
        vec![],
        0,
    );
    insert_at(ctx, cast_op, block, prev_op);
    MirCastOp::new(cast_op).set_attr_cast_kind(ctx, MirCastKindAttr::PtrToPtr);
    let cast_value = cast_op.deref(ctx).get_result(0);
    (cast_value, Some(cast_op))
}

/// Recover the pointee (element) type of a slot value. Panics if the value is
/// not a `MirPtrType`; this invariant is established when a slot is recorded
/// via [`ValueMap::set_slot`] after an [`ValueMap::emit_alloca`] call.
fn slot_pointee(ctx: &Context, slot: Value) -> Ptr<TypeObj> {
    let ptr_ty = slot.get_type(ctx);
    ptr_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .expect("ValueMap slot must carry a MirPtrType value")
        .pointee
}

/// Insert `op` after `prev_op` if provided, else at the front of `block`.
fn insert_at(
    ctx: &mut Context,
    op: Ptr<Operation>,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
) {
    match prev_op {
        Some(prev) => op.insert_after(ctx, prev),
        None => op.insert_at_front(block, ctx),
    }
}

// =============================================================================
// Slot address-space inference
// =============================================================================

/// Per-local inferred address space for the alloca slot's *pointee* type.
///
/// Only meaningful when the local's translated type is itself a pointer
/// (`MirPtrType`). For non-pointer locals this state is computed but never
/// consulted — the slot pointee has no addrspace field to override.
///
/// The lattice is monotone (`Uninit → Known(n) → Generic`, never backwards);
/// this guarantees the fixed-point in [`SlotAddrSpaceMap::analyze`] terminates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SlotAddrSpace {
    /// No classified writes observed yet. The slot will fall back to the
    /// Rust-declared address space in [`SlotAddrSpaceMap::effective`]. This
    /// is the critical "trust the type" state: if the classifier has
    /// nothing confident to say, we defer to Rust's typed addrspace rather
    /// than demoting to generic.
    Uninit,
    /// Every classified write so far produced a pointer in this address
    /// space; the slot can safely be typed to match.
    Known(u32),
    /// Classified writes from multiple, disagreeing address spaces were
    /// observed. The slot must stay generic (`addrspace(0)`) so
    /// `maybe_ptr_coerce` can cast every store site to match.
    Generic,
}

impl SlotAddrSpace {
    /// Monotone join of two observations on the same slot.
    ///
    /// - `Uninit` is the identity (no observation ≡ no change).
    /// - Two `Known(n)` that agree stay `Known(n)`.
    /// - Two disagreeing `Known(_)` collapse to `Generic`.
    /// - `Generic` is the absorbing element (once demoted, stay demoted).
    fn merge(self, other: SlotAddrSpace) -> SlotAddrSpace {
        use SlotAddrSpace::*;
        match (self, other) {
            (Uninit, x) | (x, Uninit) => x,
            (Generic, _) | (_, Generic) => Generic,
            (Known(a), Known(b)) if a == b => Known(a),
            (Known(_), Known(_)) => Generic,
        }
    }
}

/// Result of classifying a single write's right-hand side.
///
/// Fed into [`SlotAddrSpace::merge`] by the analyzer driver.
#[derive(Debug, Clone, Copy)]
enum WriteClass {
    /// We are confident this write produced a pointer in this address space.
    /// Merging promotes an `Uninit` slot to `Known(n)` and disagreeing
    /// `Known(_)` slots to `Generic`.
    Classified(u32),
    /// The write produced something we deliberately don't reason about
    /// (aggregates, arithmetic, casts, complex projections, `Ref`/`AddressOf`,
    /// arbitrary function returns not in the intrinsic whitelist, …).
    ///
    /// Crucially this does **not** demote the slot: we already trust
    /// Rust's declared addrspace as the fallback (see [`SlotAddrSpace::Uninit`]),
    /// so an unclassified write is equivalent to no observation. If the slot
    /// was already `Known(n)` from a prior classified write, it stays
    /// `Known(n)`. Demoting would be catastrophic for locals whose declared
    /// type is a non-generic addrspace (e.g. `&mut SharedArray<_>` reborrows
    /// in `tiled_gemm`).
    Unclassified,
    /// The write is `_y = _x`-style propagation from a local whose state is
    /// still [`SlotAddrSpace::Uninit`]. That's a timing artefact of the
    /// fixed-point iteration, not a genuine "unknown" — we skip this write
    /// and re-examine it on the next iteration, by which point the source
    /// local will have either classified or stayed `Uninit`.
    Pending,
}

/// Per-local result of the address-space pre-scan.
///
/// Indexed by [`mir::Local`]; `body::emit_entry_allocas` consults this once
/// per non-ZST local to decide the alloca pointee's addrspace.
pub struct SlotAddrSpaceMap {
    classes: Vec<SlotAddrSpace>,
}

impl SlotAddrSpaceMap {
    /// Infer per-local slot pointee address spaces by pre-scanning `body`.
    ///
    /// Each iteration walks every statement and every `Call` terminator,
    /// classifies the RHS, and merges classified observations into the
    /// destination local's state. Only `Classified(_)` observations change
    /// state; `Unclassified` and `Pending` are no-ops by design (see
    /// `WriteClass::Unclassified` / `resolve` for the rationale).
    ///
    /// Convergence: each local can transition at most
    /// `Uninit → Known(n) → Generic` (two steps). Propagation chains
    /// `_a = _b = … = _z` are bounded by `num_locals`, so `num_locals + 2`
    /// iterations are guaranteed sufficient.
    pub fn analyze(body: &mir::Body) -> Self {
        let num_locals = body.locals().len();
        let mut classes = vec![SlotAddrSpace::Uninit; num_locals];

        let cap = num_locals.saturating_add(2).max(2);
        for _ in 0..cap {
            let mut changed = false;

            for block in &body.blocks {
                for stmt in &block.statements {
                    let mir::StatementKind::Assign(place, rvalue) = &stmt.kind else {
                        continue;
                    };
                    if !place.projection.is_empty() {
                        continue;
                    }
                    let class = classify_rvalue(rvalue, &classes);
                    if let Some(observation) = resolve(class, false)
                        && merge_into(&mut classes, place.local, observation)
                    {
                        changed = true;
                    }
                }

                let mir::TerminatorKind::Call {
                    func, destination, ..
                } = &block.terminator.kind
                else {
                    continue;
                };
                if !destination.projection.is_empty() {
                    continue;
                }
                let class = classify_call(func);
                if let Some(observation) = resolve(class, false)
                    && merge_into(&mut classes, destination.local, observation)
                {
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        Self { classes }
    }

    /// Effective address space for `local`'s slot pointee.
    ///
    /// - `Known(n)` → `n` (the inferred addrspace).
    /// - `Generic`  → `address_space::GENERIC` (writes disagreed or were
    ///   unclassified).
    /// - `Uninit`   → `rust_declared` (no classified writes seen; keep
    ///   whatever `translate_type` produced).
    pub fn effective(&self, local: mir::Local, rust_declared: u32) -> u32 {
        match self
            .classes
            .get(local)
            .copied()
            .unwrap_or(SlotAddrSpace::Uninit)
        {
            SlotAddrSpace::Uninit => rust_declared,
            SlotAddrSpace::Known(n) => n,
            SlotAddrSpace::Generic => address_space::GENERIC,
        }
    }
}

/// Turn a [`WriteClass`] into a [`SlotAddrSpace`] observation, or `None`
/// when the observation should be skipped this iteration.
///
/// `Unclassified` and `Pending` both resolve to `None` ("skip"): we never
/// demote a slot based on lack of information. Demotion to `Generic` only
/// happens when two genuinely disagreeing `Classified(_)` observations hit
/// the same slot.
fn resolve(class: WriteClass, _final_pass: bool) -> Option<SlotAddrSpace> {
    match class {
        WriteClass::Classified(n) => Some(SlotAddrSpace::Known(n)),
        WriteClass::Unclassified | WriteClass::Pending => None,
    }
}

/// Merge `observation` into `classes[local]`. Returns `true` if the slot's
/// state changed.
fn merge_into(
    classes: &mut [SlotAddrSpace],
    local: mir::Local,
    observation: SlotAddrSpace,
) -> bool {
    let Some(slot) = classes.get_mut(local) else {
        return false;
    };
    let merged = slot.merge(observation);
    if merged != *slot {
        *slot = merged;
        true
    } else {
        false
    }
}

/// Classify the write produced by an `Assign(_, rvalue)` statement.
///
/// The rule set is intentionally narrow: when in doubt, return
/// [`WriteClass::Unclassified`] so the final state merges to `Generic`. The
/// safety invariant is "a slot is only promoted out of generic when every
/// write into it is confidently classified," so an incomplete classifier
/// at worst leaves the slot generic — i.e. today's behavior.
fn classify_rvalue(rvalue: &mir::Rvalue, classes: &[SlotAddrSpace]) -> WriteClass {
    match rvalue {
        // `_y = _x` — propagate `_x`'s current classification. `Move` and
        // `Copy` are indistinguishable for addrspace purposes. `CopyForDeref`
        // behaves the same at this layer.
        mir::Rvalue::Use(mir::Operand::Copy(place))
        | mir::Rvalue::Use(mir::Operand::Move(place))
        | mir::Rvalue::CopyForDeref(place)
            if place.projection.is_empty() =>
        {
            propagate_from_local(place.local, classes)
        }
        // `_y = CONSTANT` — a constant-operand pointer to a shared-memory
        // static (e.g. `&mut TILE_A` where `TILE_A: SharedArray<...>`)
        // lowers to a `mir.shared_alloc` in `translate_operand` whose
        // result is `addrspace(3)`. The matching `WriteClass::Classified`
        // here keeps the destination slot typed to match, avoiding the
        // otherwise-inevitable `PtrToPtr` narrow-to-generic cast.
        mir::Rvalue::Use(mir::Operand::Constant(const_op)) => classify_constant(const_op),
        // Every other rvalue shape (aggregates, arithmetic, casts, complex
        // projections, `Ref`/`AddressOf`, …) we decline to reason about
        // here. The matching `Call`-terminator classifier handles the
        // pointer-producing intrinsics; the remaining cases can be
        // tightened in a follow-up if a benchmark asks for it.
        _ => WriteClass::Unclassified,
    }
}

/// Classify a constant operand's address space.
///
/// Kept in sync with `rvalue::is_shared_array_pointer` /
/// `is_barrier_pointer` / ordinary static handling — those gate the emitter;
/// this gates the slot address-space classifier.
fn classify_constant(const_op: &mir::ConstOperand) -> WriteClass {
    let ty = const_op.const_.ty();
    let TyKind::RigidTy(RigidTy::RawPtr(pointee, _) | RigidTy::Ref(_, pointee, _)) = ty.kind()
    else {
        return WriteClass::Unclassified;
    };

    if let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = pointee.kind()
        && matches!(adt_def.trimmed_name().as_str(), "SharedArray" | "Barrier")
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    let ConstantKind::Allocated(alloc) = const_op.const_.kind() else {
        return WriteClass::Unclassified;
    };
    if alloc.is_null().unwrap_or(false) {
        return WriteClass::Unclassified;
    }
    let Some((_, prov)) = alloc.provenance.ptrs.first() else {
        return WriteClass::Unclassified;
    };
    match GlobalAlloc::from(prov.0) {
        GlobalAlloc::Static(static_def) => {
            // `#[constant]` statics live in addrspace(4) and are recognised
            // by the `ConstantMemory<T>` wrapper on the static's declared type.
            // Other statics live in addrspace(1).
            let static_ty = static_def.ty();
            if is_constant_wrapper_type(&static_ty) {
                WriteClass::Classified(address_space::CONSTANT)
            } else {
                WriteClass::Classified(address_space::GLOBAL)
            }
        }
        _ => WriteClass::Unclassified,
    }
}

/// `true` if `ty` is `cuda_device::ConstantMemory<_>`. Detection by trimmed ADT
/// name, mirroring the `SharedArray | Barrier` check above in
/// [`classify_constant`].
pub(super) fn is_constant_wrapper_type(ty: &rustc_public::ty::Ty) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};
    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = ty.kind() else {
        return false;
    };
    adt_def.krate().name.as_str() == "cuda_device"
        && adt_def.trimmed_name().as_str() == "ConstantMemory"
}

/// Classify the write produced by a `Call` terminator's destination.
///
/// Mirrors the intrinsic dispatch table in
/// `translator/terminator/mod.rs::try_dispatch_intrinsic`. Any intrinsic
/// whose emitter unconditionally produces a pointer in a specific address
/// space is listed here; new intrinsics should add an entry on the same
/// commit that adds their emitter.
fn classify_call(func: &mir::Operand) -> WriteClass {
    let mir::Operand::Constant(const_op) = func else {
        return WriteClass::Unclassified;
    };
    if !matches!(const_op.const_.kind(), ConstantKind::ZeroSized) {
        return WriteClass::Unclassified;
    }
    let TyKind::RigidTy(RigidTy::FnDef(fn_def, substs)) = const_op.const_.ty().kind() else {
        return WriteClass::Unclassified;
    };
    let path = fn_def.name();
    let substs_str = format!("{substs:?}");
    let on_shared_array = substs_str.contains("SharedArray");

    // --- addrspace 3 (shared) producers -------------------------------------
    //
    // `SharedArray::index` / `IndexMut::index_mut` on a `SharedArray<T, N>`
    // lower to `emit_shared_array_index`, which offsets the shared-memory
    // base pointer and returns `*mut T addrspace(3)`.
    if on_shared_array
        && matches!(
            path.as_str(),
            "std::ops::Index::index"
                | "core::ops::Index::index"
                | "std::ops::IndexMut::index_mut"
                | "core::ops::IndexMut::index_mut"
        )
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    // `DynamicSharedArray::<T, ALIGN>::{get, get_raw, offset}` all hand back
    // pointers into the extern-shared region (`addrspace(3)`).
    if path.contains("DynamicSharedArray") && (path.contains("::get") || path.contains("::offset"))
    {
        return WriteClass::Classified(address_space::SHARED);
    }

    // --- explicit narrow to generic -----------------------------------------
    //
    // `SharedArray::as_ptr` / `as_mut_ptr` deliberately `cvta.shared` the
    // base pointer into the generic address space, so the callee sees
    // `addrspace(0)`.
    if path.contains("SharedArray") && (path.contains("as_ptr") || path.contains("as_mut_ptr")) {
        return WriteClass::Classified(address_space::GENERIC);
    }

    WriteClass::Unclassified
}

/// Inherit classification from a source local (for `_y = _x` chains).
fn propagate_from_local(local: mir::Local, classes: &[SlotAddrSpace]) -> WriteClass {
    match classes.get(local).copied().unwrap_or(SlotAddrSpace::Uninit) {
        SlotAddrSpace::Known(n) => WriteClass::Classified(n),
        SlotAddrSpace::Generic => WriteClass::Unclassified,
        // Source hasn't been classified yet in this iteration — try again
        // on the next pass rather than prematurely demoting the destination.
        SlotAddrSpace::Uninit => WriteClass::Pending,
    }
}

/// If `elem_ty` is a `MirPtrType`, return it with `target` replacing the
/// current address space; otherwise return `elem_ty` unchanged.
///
/// Used by `body::emit_entry_allocas` to override a Rust-declared pointer
/// addrspace with the one inferred by [`SlotAddrSpaceMap`].
pub fn align_pointer_addr_space(
    ctx: &mut Context,
    elem_ty: Ptr<TypeObj>,
    target: u32,
) -> Ptr<TypeObj> {
    let ptr_info = elem_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|pt| (pt.pointee, pt.is_mutable, pt.address_space));
    let Some((pointee, is_mutable, current)) = ptr_info else {
        return elem_ty;
    };
    if current == target {
        return elem_ty;
    }
    MirPtrType::get(ctx, pointee, is_mutable, target).into()
}

/// Extract a pointer type's address space, or `None` if `elem_ty` is not a
/// [`MirPtrType`]. Useful as the `rust_declared` fallback for
/// [`SlotAddrSpaceMap::effective`].
pub fn pointer_addr_space(ctx: &Context, elem_ty: Ptr<TypeObj>) -> Option<u32> {
    elem_ty
        .deref(ctx)
        .downcast_ref::<MirPtrType>()
        .map(|pt| pt.address_space)
}
