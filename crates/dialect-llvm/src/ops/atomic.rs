/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM atomic operations: atomicrmw, cmpxchg, atomic load/store, fence.
//!
//! These ops produce textual LLVM IR atomic instructions. Each carries
//! ordering and syncscope attributes.
//!
//! # Op Interface: `LlvmAtomicOpInterface`
//!
//! All five ops implement [`LlvmAtomicOpInterface`], providing uniform
//! access to `ordering()` and `syncscope()`. The export code uses this
//! to share syncscope/ordering formatting logic.

use pliron::{
    builtin::op_interfaces::{
        NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{def_op, op_interface, op_interface_impl},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{derive_attr_get_set, derive_op_interface_impl, format_op, verify_succ};

use crate::attributes::{LlvmAtomicOrdering, LlvmAtomicRmwKind, LlvmSyncScope};
use crate::types::PointerType;

// =============================================================================
// Op Interface
// =============================================================================

/// Shared interface for all LLVM atomic operations.
///
/// Provides uniform access to ordering and syncscope so that export.rs
/// can format these through shared helpers rather than duplicating logic.
#[op_interface]
pub trait LlvmAtomicOpInterface {
    /// The memory ordering for this atomic operation.
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering;

    /// The syncscope for this atomic operation.
    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope;

    fn verify(_op: &dyn Op, _ctx: &Context) -> pliron::result::Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

// =============================================================================
// Formatting helpers (used by export.rs)
// =============================================================================

/// Format a syncscope for LLVM IR text output.
///
/// Returns the string to insert before the ordering keyword:
/// - Device -> ` syncscope("device")`
/// - Block  -> ` syncscope("block")`
/// - System -> `` (empty -- system is the default)
pub fn format_syncscope(scope: &LlvmSyncScope) -> &'static str {
    match scope {
        LlvmSyncScope::Device => " syncscope(\"device\")",
        LlvmSyncScope::Block => " syncscope(\"block\")",
        LlvmSyncScope::System => "",
    }
}

/// Format an ordering for LLVM IR text output.
pub fn format_ordering(ord: &LlvmAtomicOrdering) -> &'static str {
    match ord {
        LlvmAtomicOrdering::Monotonic => "monotonic",
        LlvmAtomicOrdering::Acquire => "acquire",
        LlvmAtomicOrdering::Release => "release",
        LlvmAtomicOrdering::AcqRel => "acq_rel",
        LlvmAtomicOrdering::SeqCst => "seq_cst",
    }
}

/// Format an atomicrmw operation kind for LLVM IR text output.
pub fn format_rmw_kind(kind: &LlvmAtomicRmwKind) -> &'static str {
    match kind {
        LlvmAtomicRmwKind::Add => "add",
        LlvmAtomicRmwKind::Sub => "sub",
        LlvmAtomicRmwKind::Xchg => "xchg",
        LlvmAtomicRmwKind::And => "and",
        LlvmAtomicRmwKind::Or => "or",
        LlvmAtomicRmwKind::Xor => "xor",
        LlvmAtomicRmwKind::Max => "max",
        LlvmAtomicRmwKind::Min => "min",
        LlvmAtomicRmwKind::UMax => "umax",
        LlvmAtomicRmwKind::UMin => "umin",
        LlvmAtomicRmwKind::FAdd => "fadd",
    }
}

// =============================================================================
// AtomicRmwOp
// =============================================================================

/// LLVM `atomicrmw` instruction.
///
/// ```llvm
/// %old = atomicrmw add ptr %p, i32 %v syncscope("device") monotonic
/// ```
///
/// # Operands
/// - 0: `ptr` -- pointer to target
/// - 1: `val` -- value for the operation
///
/// # Results
/// - 0: the old value before modification
///
/// # Attributes
/// - `rmw_kind`: Add, Sub, Xchg, And, Or, Xor, ...
/// - `ordering`: Monotonic, Acquire, Release, AcqRel, SeqCst
/// - `syncscope`: System, Device, Block
#[format_op]
#[def_op("llvm.atomicrmw")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<2>)]
#[derive_attr_get_set(
    llvm_rmw_ordering: LlvmAtomicOrdering,
    llvm_rmw_syncscope: LlvmSyncScope,
    llvm_rmw_kind: LlvmAtomicRmwKind
)]
pub struct AtomicRmwOp;

impl AtomicRmwOp {
    /// Create a new `atomicrmw` op.
    pub fn new(
        ctx: &mut Context,
        ptr: Value,
        val: Value,
        result_ty: Ptr<pliron::r#type::TypeObj>,
        rmw_kind: LlvmAtomicRmwKind,
        ordering: LlvmAtomicOrdering,
        syncscope: LlvmSyncScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, val],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_llvm_rmw_kind(ctx, rmw_kind);
        this.set_attr_llvm_rmw_ordering(ctx, ordering);
        this.set_attr_llvm_rmw_syncscope(ctx, syncscope);
        this
    }

    /// Get the RMW operation kind.
    pub fn rmw_kind(&self, ctx: &Context) -> LlvmAtomicRmwKind {
        self.get_attr_llvm_rmw_kind(ctx)
            .expect("AtomicRmwOp missing rmw_kind")
            .clone()
    }
}

impl Verify for AtomicRmwOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let ptr_ty = op.get_operand(0).get_type(ctx);
        if ptr_ty.deref(ctx).downcast_ref::<PointerType>().is_none() {
            return verify_err!(op.loc(), "llvm.atomicrmw first operand must be a pointer");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl LlvmAtomicOpInterface for AtomicRmwOp {
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_rmw_ordering(ctx)
            .expect("AtomicRmwOp missing ordering")
            .clone()
    }

    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope {
        self.get_attr_llvm_rmw_syncscope(ctx)
            .expect("AtomicRmwOp missing syncscope")
            .clone()
    }
}

// =============================================================================
// AtomicCmpxchgOp
// =============================================================================

/// LLVM `cmpxchg` instruction.
///
/// ```llvm
/// %result = cmpxchg ptr %p, i32 %cmp, i32 %new syncscope("device") acq_rel acquire
/// ```
///
/// LLVM returns `{ T, i1 }` but we model only the `T` result. The success
/// boolean is extracted via `extractvalue` in the export.
///
/// # Operands
/// - 0: `ptr` -- pointer to target
/// - 1: `cmp` -- expected value
/// - 2: `new` -- value to store on match
///
/// # Results
/// - 0: the old value at `*ptr`
///
/// # Attributes
/// - `success_ordering`, `failure_ordering`, `syncscope`
#[format_op]
#[def_op("llvm.cmpxchg")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<3>)]
#[derive_attr_get_set(
    llvm_cas_success_ordering: LlvmAtomicOrdering,
    llvm_cas_failure_ordering: LlvmAtomicOrdering,
    llvm_cas_syncscope: LlvmSyncScope
)]
pub struct AtomicCmpxchgOp;

impl AtomicCmpxchgOp {
    /// Create a new `cmpxchg` op.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        ctx: &mut Context,
        ptr: Value,
        cmp: Value,
        new: Value,
        result_ty: Ptr<pliron::r#type::TypeObj>,
        success_ordering: LlvmAtomicOrdering,
        failure_ordering: LlvmAtomicOrdering,
        syncscope: LlvmSyncScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, cmp, new],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_llvm_cas_success_ordering(ctx, success_ordering);
        this.set_attr_llvm_cas_failure_ordering(ctx, failure_ordering);
        this.set_attr_llvm_cas_syncscope(ctx, syncscope);
        this
    }

    /// Get the success ordering.
    pub fn success_ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_cas_success_ordering(ctx)
            .expect("AtomicCmpxchgOp missing success_ordering")
            .clone()
    }

    /// Get the failure ordering.
    pub fn failure_ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_cas_failure_ordering(ctx)
            .expect("AtomicCmpxchgOp missing failure_ordering")
            .clone()
    }
}

impl Verify for AtomicCmpxchgOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let ptr_ty = op.get_operand(0).get_type(ctx);
        if ptr_ty.deref(ctx).downcast_ref::<PointerType>().is_none() {
            return verify_err!(op.loc(), "llvm.cmpxchg first operand must be a pointer");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl LlvmAtomicOpInterface for AtomicCmpxchgOp {
    /// Returns the **success** ordering.
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.success_ordering(ctx)
    }

    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope {
        self.get_attr_llvm_cas_syncscope(ctx)
            .expect("AtomicCmpxchgOp missing syncscope")
            .clone()
    }
}

// =============================================================================
// AtomicLoadOp
// =============================================================================

/// LLVM atomic `load` instruction.
///
/// ```llvm
/// %val = load atomic i32, ptr %p syncscope("device") acquire
/// ```
///
/// # Operands
/// - 0: `ptr` -- pointer to load from
///
/// # Results
/// - 0: the loaded value
///
/// # Attributes
/// - `ordering`: Monotonic, Acquire, SeqCst
/// - `syncscope`: System, Device, Block
#[format_op]
#[def_op("llvm.atomic_load")]
#[derive_op_interface_impl(NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface)]
#[derive_attr_get_set(llvm_ld_ordering: LlvmAtomicOrdering, llvm_ld_syncscope: LlvmSyncScope)]
pub struct AtomicLoadOp;

impl AtomicLoadOp {
    /// Create a new atomic load op.
    pub fn new(
        ctx: &mut Context,
        ptr: Value,
        result_ty: Ptr<pliron::r#type::TypeObj>,
        ordering: LlvmAtomicOrdering,
        syncscope: LlvmSyncScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_llvm_ld_ordering(ctx, ordering);
        this.set_attr_llvm_ld_syncscope(ctx, syncscope);
        this
    }
}

impl Verify for AtomicLoadOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let ptr_ty = op.get_operand(0).get_type(ctx);
        if ptr_ty.deref(ctx).downcast_ref::<PointerType>().is_none() {
            return verify_err!(op.loc(), "llvm.atomic_load operand must be a pointer");
        }
        Ok(())
    }
}

#[op_interface_impl]
impl LlvmAtomicOpInterface for AtomicLoadOp {
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_ld_ordering(ctx)
            .expect("AtomicLoadOp missing ordering")
            .clone()
    }

    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope {
        self.get_attr_llvm_ld_syncscope(ctx)
            .expect("AtomicLoadOp missing syncscope")
            .clone()
    }
}

// =============================================================================
// AtomicStoreOp
// =============================================================================

/// LLVM atomic `store` instruction.
///
/// ```llvm
/// store atomic i32 %v, ptr %p syncscope("device") release
/// ```
///
/// # Operands
/// - 0: `val` -- value to store
/// - 1: `ptr` -- pointer to store to
///
/// # Results
/// None.
///
/// # Attributes
/// - `ordering`: Monotonic, Release, SeqCst
/// - `syncscope`: System, Device, Block
#[format_op]
#[def_op("llvm.atomic_store")]
#[derive_op_interface_impl(NResultsInterface<0>, NOpdsInterface<2>)]
#[derive_attr_get_set(llvm_st_ordering: LlvmAtomicOrdering, llvm_st_syncscope: LlvmSyncScope)]
pub struct AtomicStoreOp;

impl AtomicStoreOp {
    /// Create a new atomic store op.
    pub fn new(
        ctx: &mut Context,
        val: Value,
        ptr: Value,
        ordering: LlvmAtomicOrdering,
        syncscope: LlvmSyncScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![val, ptr],
            vec![],
            0,
        );
        let this = Self { op };
        this.set_attr_llvm_st_ordering(ctx, ordering);
        this.set_attr_llvm_st_syncscope(ctx, syncscope);
        this
    }

    /// Get the value operand.
    pub fn value_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the pointer operand.
    pub fn address_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

impl Verify for AtomicStoreOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let ptr_ty = op.get_operand(1).get_type(ctx);
        if ptr_ty.deref(ctx).downcast_ref::<PointerType>().is_none() {
            return verify_err!(
                op.loc(),
                "llvm.atomic_store second operand must be a pointer"
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl LlvmAtomicOpInterface for AtomicStoreOp {
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_st_ordering(ctx)
            .expect("AtomicStoreOp missing ordering")
            .clone()
    }

    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope {
        self.get_attr_llvm_st_syncscope(ctx)
            .expect("AtomicStoreOp missing syncscope")
            .clone()
    }
}

// =============================================================================
// FenceOp
// =============================================================================

/// LLVM `fence` instruction.
///
/// ```llvm
/// fence syncscope("device") release
/// ```
///
/// # Operands
/// None.
///
/// # Results
/// None.
///
/// # Attributes
/// - `ordering`: Acquire, Release, AcqRel, SeqCst
/// - `syncscope`: System, Device, Block
#[format_op]
#[def_op("llvm.fence")]
#[derive_op_interface_impl(NResultsInterface<0>, NOpdsInterface<0>)]
#[derive_attr_get_set(llvm_fence_ordering: LlvmAtomicOrdering, llvm_fence_syncscope: LlvmSyncScope)]
#[verify_succ]
pub struct FenceOp;

impl FenceOp {
    /// Create a new fence op.
    pub fn new(ctx: &mut Context, ordering: LlvmAtomicOrdering, syncscope: LlvmSyncScope) -> Self {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);
        let this = Self { op };
        this.set_attr_llvm_fence_ordering(ctx, ordering);
        this.set_attr_llvm_fence_syncscope(ctx, syncscope);
        this
    }
}

#[op_interface_impl]
impl LlvmAtomicOpInterface for FenceOp {
    fn ordering(&self, ctx: &Context) -> LlvmAtomicOrdering {
        self.get_attr_llvm_fence_ordering(ctx)
            .expect("FenceOp missing ordering")
            .clone()
    }

    fn syncscope(&self, ctx: &Context) -> LlvmSyncScope {
        self.get_attr_llvm_fence_syncscope(ctx)
            .expect("FenceOp missing syncscope")
            .clone()
    }
}

// =============================================================================
// Registration
// =============================================================================

/// Register all LLVM atomic operations.
pub fn register(ctx: &mut Context) {
    AtomicRmwOp::register(ctx);
    AtomicCmpxchgOp::register(ctx);
    AtomicLoadOp::register(ctx);
    AtomicStoreOp::register(ctx);
    FenceOp::register(ctx);
}
