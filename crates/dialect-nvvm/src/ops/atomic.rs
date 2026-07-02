/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Atomic operations for GPU memory.
//!
//! These operations represent atomic reads, writes, and read-modify-writes on
//! GPU memory. They carry **ordering** and **scope** attributes so that
//! downstream lowering can emit the correct LLVM IR (and ultimately PTX).
//!
//! # Operations
//!
//! ```text
//! ┌─────────────────────────┬─────────────────────────────────────────────────┐
//! │ Op                      │ LLVM IR it lowers to                            │
//! ├─────────────────────────┼─────────────────────────────────────────────────┤
//! │ NvvmAtomicLoadOp        │ load atomic <ty>, ptr %p syncscope(...) <ord>   │
//! │ NvvmAtomicStoreOp       │ store atomic <ty> %v, ptr %p syncscope(...) ... │
//! │ NvvmAtomicRmwOp         │ atomicrmw <op> ptr %p, <ty> %v syncscope(...)   │
//! │ NvvmAtomicCmpxchgOp     │ cmpxchg ptr %p, <ty> %cmp, <ty> %new ...       │
//! └─────────────────────────┴─────────────────────────────────────────────────┘
//! ```
//!
//! # Op Interface: `NvvmAtomicOpInterface`
//!
//! All four ops implement [`NvvmAtomicOpInterface`], which provides uniform
//! access to `ordering()`, `scope()`, and `ptr_operand()`. This lets
//! mir-lower handle ordering-to-fence and scope-to-syncscope mapping through
//! one code path.

use dialect_mir::types::{MirPtrType, address_space};
use pliron::{
    attribute::Attribute,
    builtin::op_interfaces::{
        NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface,
    },
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::{op_interface, op_interface_impl},
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    value::Value,
    verify_err,
};
use pliron_derive::{pliron_attr, pliron_op};

// =============================================================================
// Attribute Enums
// =============================================================================

/// Memory ordering for atomic operations.
///
/// Maps to LLVM orderings and ultimately to PTX ordering qualifiers:
///
/// | Variant   | LLVM         | PTX          |
/// |-----------|--------------|--------------|
/// | Relaxed   | `monotonic`  | `.relaxed`   |
/// | Acquire   | `acquire`    | `.acquire`   |
/// | Release   | `release`    | `.release`   |
/// | AcqRel    | `acq_rel`    | `.acq_rel`   |
/// | SeqCst    | `seq_cst`    | `fence.sc +` |
#[pliron_attr(name = "nvvm.atomic_ordering", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicOrdering {
    Relaxed,
    Acquire,
    Release,
    AcqRel,
    SeqCst,
}

/// Scope of the atomic operation -- which threads observe it.
///
/// | Variant | LLVM syncscope    | PTX    |
/// |---------|-------------------|--------|
/// | Device  | `"device"`        | `.gpu` |
/// | Block   | `"block"`         | `.cta` |
/// | System  | (default / `""`)  | `.sys` |
#[pliron_attr(name = "nvvm.atomic_scope", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicScope {
    Device,
    Block,
    System,
}

/// Kind of read-modify-write operation.
///
/// These map 1:1 to LLVM `atomicrmw` operation keywords.
#[pliron_attr(name = "nvvm.atomic_rmw_kind", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub enum AtomicRmwKind {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
    Min,
    Max,
    UMin,
    UMax,
    FAdd,
}

// =============================================================================
// Op Interface
// =============================================================================

/// Shared interface for all NVVM atomic operations.
///
/// Provides uniform access to ordering and scope so that mir-lower can
/// handle the ordering->fence and scope->syncscope mapping through a
/// single code path, regardless of which atomic op it is.
#[op_interface]
pub trait NvvmAtomicOpInterface {
    /// The memory ordering for this atomic operation.
    fn ordering(&self, ctx: &Context) -> AtomicOrdering;

    /// The scope (which threads observe the atomic).
    fn scope(&self, ctx: &Context) -> AtomicScope;

    /// The pointer operand (always the first operand).
    fn ptr_operand(&self, ctx: &Context) -> Value;

    fn verify(_op: &dyn Op, _ctx: &Context) -> pliron::result::Result<()>
    where
        Self: Sized,
    {
        // Structural verification is done by each op's own Verify impl.
        Ok(())
    }
}

// =============================================================================
// NvvmAtomicLoadOp
// =============================================================================

/// Atomic load from GPU memory.
///
/// # Operands
///
/// - `ptr`: pointer to the value to load
///
/// # Results
///
/// - loaded value (i32, i64, f32, f64)
///
/// # Attributes
///
/// - `ordering`: `Relaxed`, `Acquire`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_load",
    format,
    verifier = "succ",
    interfaces = [NResultsInterface<1>, OneResultInterface, NOpdsInterface<1>, OneOpdInterface],
    attributes = (nvvm_ld_ordering: AtomicOrdering, nvvm_ld_scope: AtomicScope)
)]
pub struct NvvmAtomicLoadOp;

impl NvvmAtomicLoadOp {
    /// Create a new atomic load op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicLoadOp { op }
    }

    /// Create a new atomic load from scratch.
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        result_ty: pliron::r#type::TypeHandle,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr],
            vec![],
            0,
        );
        let this = NvvmAtomicLoadOp { op };
        this.set_attr_nvvm_ld_ordering(ctx, ordering);
        this.set_attr_nvvm_ld_scope(ctx, scope);
        this
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicLoadOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_ld_ordering(ctx)
            .expect("NvvmAtomicLoadOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_ld_scope(ctx)
            .expect("NvvmAtomicLoadOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// NvvmAtomicStoreOp
// =============================================================================

/// Atomic store to GPU memory.
///
/// # Operands
///
/// - `val`: value to store
/// - `ptr`: pointer to store to
///
/// # Results
///
/// None.
///
/// # Attributes
///
/// - `ordering`: `Relaxed`, `Release`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_store",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<0>],
    attributes = (nvvm_st_ordering: AtomicOrdering, nvvm_st_scope: AtomicScope)
)]
pub struct NvvmAtomicStoreOp;

impl NvvmAtomicStoreOp {
    /// Create a new atomic store op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicStoreOp { op }
    }

    /// Create a new atomic store from scratch.
    pub fn build(
        ctx: &mut Context,
        val: Value,
        ptr: Value,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![val, ptr],
            vec![],
            0,
        );
        let this = NvvmAtomicStoreOp { op };
        this.set_attr_nvvm_st_ordering(ctx, ordering);
        this.set_attr_nvvm_st_scope(ctx, scope);
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

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicStoreOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_st_ordering(ctx)
            .expect("NvvmAtomicStoreOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_st_scope(ctx)
            .expect("NvvmAtomicStoreOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

// =============================================================================
// NvvmAtomicRmwOp
// =============================================================================

/// Atomic read-modify-write on GPU memory.
///
/// Returns the **previous** value at the memory location.
///
/// # Operands
///
/// - `ptr`: pointer to the target
/// - `val`: value to combine with the target
///
/// # Results
///
/// - the old value before the operation
///
/// # Attributes
///
/// - `rmw_kind`: `Add`, `Sub`, `And`, `Or`, `Xor`, `Xchg`, etc.
/// - `ordering`: `Relaxed`, `Acquire`, `Release`, `AcqRel`, or `SeqCst`
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_rmw",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        nvvm_rmw_ordering: AtomicOrdering,
        nvvm_rmw_scope: AtomicScope,
        nvvm_rmw_kind: AtomicRmwKind
    )
)]
pub struct NvvmAtomicRmwOp;

impl NvvmAtomicRmwOp {
    /// Create a new atomic RMW op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicRmwOp { op }
    }

    /// Create a new atomic RMW from scratch.
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        val: Value,
        result_ty: pliron::r#type::TypeHandle,
        rmw_kind: AtomicRmwKind,
        ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, val],
            vec![],
            0,
        );
        let this = NvvmAtomicRmwOp { op };
        this.set_attr_nvvm_rmw_kind(ctx, rmw_kind);
        this.set_attr_nvvm_rmw_ordering(ctx, ordering);
        this.set_attr_nvvm_rmw_scope(ctx, scope);
        this
    }

    /// Get the RMW operation kind.
    pub fn rmw_kind(&self, ctx: &Context) -> AtomicRmwKind {
        self.get_attr_nvvm_rmw_kind(ctx)
            .expect("NvvmAtomicRmwOp missing rmw_kind attribute")
            .clone()
    }

    /// Get the pointer operand.
    pub fn ptr_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the value operand.
    pub fn val_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicRmwOp {
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_rmw_ordering(ctx)
            .expect("NvvmAtomicRmwOp missing ordering attribute")
            .clone()
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_rmw_scope(ctx)
            .expect("NvvmAtomicRmwOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// NvvmAtomicCmpxchgOp
// =============================================================================

/// Atomic compare-and-exchange on GPU memory.
///
/// If `*ptr == cmp`, stores `new` and returns the old value. Otherwise
/// returns the current value without modifying memory.
///
/// # Operands
///
/// - `ptr`: pointer to the target
/// - `cmp`: expected value
/// - `new`: value to store if comparison succeeds
///
/// # Results
///
/// - the old value at `*ptr`
///
/// # Attributes
///
/// - `success_ordering`: ordering if comparison succeeds
/// - `failure_ordering`: ordering if comparison fails
/// - `scope`: `Device`, `Block`, or `System`
#[pliron_op(
    name = "nvvm.atomic_cmpxchg",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<3>, NResultsInterface<1>, OneResultInterface],
    attributes = (
        nvvm_cas_success_ordering: AtomicOrdering,
        nvvm_cas_failure_ordering: AtomicOrdering,
        nvvm_cas_scope: AtomicScope
    )
)]
pub struct NvvmAtomicCmpxchgOp;

impl NvvmAtomicCmpxchgOp {
    /// Create a new cmpxchg op wrapping an existing operation.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomicCmpxchgOp { op }
    }

    /// Create a new cmpxchg from scratch.
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        ctx: &mut Context,
        ptr: Value,
        cmp: Value,
        new: Value,
        result_ty: pliron::r#type::TypeHandle,
        success_ordering: AtomicOrdering,
        failure_ordering: AtomicOrdering,
        scope: AtomicScope,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty],
            vec![ptr, cmp, new],
            vec![],
            0,
        );
        let this = NvvmAtomicCmpxchgOp { op };
        this.set_attr_nvvm_cas_success_ordering(ctx, success_ordering);
        this.set_attr_nvvm_cas_failure_ordering(ctx, failure_ordering);
        this.set_attr_nvvm_cas_scope(ctx, scope);
        this
    }

    /// Get the success ordering.
    pub fn success_ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_cas_success_ordering(ctx)
            .expect("NvvmAtomicCmpxchgOp missing success_ordering")
            .clone()
    }

    /// Get the failure ordering.
    pub fn failure_ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.get_attr_nvvm_cas_failure_ordering(ctx)
            .expect("NvvmAtomicCmpxchgOp missing failure_ordering")
            .clone()
    }

    /// Get the pointer operand.
    pub fn ptr_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the comparison operand.
    pub fn cmp_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the new value operand.
    pub fn new_opd(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

#[op_interface_impl]
impl NvvmAtomicOpInterface for NvvmAtomicCmpxchgOp {
    /// For cmpxchg, `ordering()` returns the **success** ordering.
    fn ordering(&self, ctx: &Context) -> AtomicOrdering {
        self.success_ordering(ctx)
    }

    fn scope(&self, ctx: &Context) -> AtomicScope {
        self.get_attr_nvvm_cas_scope(ctx)
            .expect("NvvmAtomicCmpxchgOp missing scope attribute")
            .clone()
    }

    fn ptr_operand(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

// =============================================================================
// Packed Atomic Helpers
// =============================================================================

fn is_u32_type(ctx: &Context, ty: pliron::r#type::TypeHandle) -> bool {
    ty.deref(ctx)
        .downcast_ref::<IntegerType>()
        .is_some_and(|integer| {
            integer.width() == 32 && integer.signedness() == Signedness::Unsigned
        })
}

/// Verify the exact raw-`u32` shape used by packed atomic-add intrinsics.
fn verify_packed_atomic(ctx: &Context, op_ptr: Ptr<Operation>, op_name: &str) -> Result<(), Error> {
    let op = &*op_ptr.deref(ctx);
    if op.get_num_operands() != 2 || op.get_num_results() != 1 {
        return verify_err!(op.loc(), "{} requires 2 operands and 1 result", op_name);
    }

    let addr_ty = op.get_operand(0).get_type(ctx);
    let addr_ty_obj = addr_ty.deref(ctx);
    let Some(addr_ty) = addr_ty_obj.downcast_ref::<MirPtrType>() else {
        return verify_err!(op.loc(), "{} address must be a MIR pointer", op_name);
    };
    if !addr_ty.is_mutable() || !is_u32_type(ctx, addr_ty.pointee) {
        return verify_err!(
            op.loc(),
            "{} address must be a mutable pointer to u32",
            op_name
        );
    }
    if !matches!(
        addr_ty.address_space(),
        address_space::GENERIC | address_space::GLOBAL
    ) {
        return verify_err!(
            op.loc(),
            "{} address must be generic or global memory",
            op_name
        );
    }

    if !is_u32_type(ctx, op.get_operand(1).get_type(ctx)) {
        return verify_err!(op.loc(), "{} addend must be u32", op_name);
    }
    if !is_u32_type(ctx, op.get_result(0).get_type(ctx)) {
        return verify_err!(op.loc(), "{} result must be u32", op_name);
    }
    Ok(())
}

// =============================================================================
// NvvmAtomAddF16x2Op
// =============================================================================

/// Packed f16x2 atomic add on global memory.
///
/// Adds two packed f16 lanes element-wise with independent 16-bit atomicity.
/// The two lane operations occur in unspecified order, so the returned `u32`
/// is not guaranteed to be one coherent previous 32-bit snapshot.
///
/// Lowered to inline PTX:
/// ```ptx
/// atom.global.add.noftz.f16x2 $0, [$1], $2;
/// ```
///
/// # Operands
///
/// - `addr` (ptr): pointer to the packed f16x2 value in global memory
/// - `val`  (u32): packed f16x2 addend
///
/// # Results
///
/// - `old` (u32): the previous packed f16x2 value at `*addr`
///
/// # Verification
///
/// - `addr` must be a mutable generic/global pointer to `u32`
/// - `val` and `old` must be `u32`
#[pliron_op(
    name = "nvvm.atom_add_f16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct NvvmAtomAddF16x2Op;

impl NvvmAtomAddF16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomAddF16x2Op { op }
    }
}

impl Verify for NvvmAtomAddF16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_packed_atomic(ctx, self.get_operation(), "nvvm.atom_add_f16x2")
    }
}

// =============================================================================
// NvvmAtomAddBf16x2Op
// =============================================================================

/// Packed bf16x2 atomic add on global memory.
///
/// Adds two packed bf16 lanes element-wise with independent 16-bit atomicity.
/// The two lane operations occur in unspecified order, so the returned `u32`
/// is not guaranteed to be one coherent previous 32-bit snapshot.
///
/// Lowered to inline PTX:
/// ```ptx
/// atom.global.add.noftz.bf16x2 $0, [$1], $2;
/// ```
///
/// # Operands
///
/// - `addr` (ptr): pointer to the packed bf16x2 value in global memory
/// - `val`  (u32): packed bf16x2 addend
///
/// # Results
///
/// - `old` (u32): the previous packed bf16x2 value at `*addr`
///
/// # Verification
///
/// - `addr` must be a mutable generic/global pointer to `u32`
/// - `val` and `old` must be `u32`
#[pliron_op(
    name = "nvvm.atom_add_bf16x2",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct NvvmAtomAddBf16x2Op;

impl NvvmAtomAddBf16x2Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        NvvmAtomAddBf16x2Op { op }
    }
}

impl Verify for NvvmAtomAddBf16x2Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        verify_packed_atomic(ctx, self.get_operation(), "nvvm.atom_add_bf16x2")
    }
}

// =============================================================================
// Registration
// =============================================================================

/// Register all atomic operations and attributes with the context.
pub fn register(ctx: &mut Context) {
    // Register attributes
    AtomicOrdering::register(ctx);
    AtomicScope::register(ctx);
    AtomicRmwKind::register(ctx);

    // Register ops
    NvvmAtomicLoadOp::register(ctx);
    NvvmAtomicStoreOp::register(ctx);
    NvvmAtomicRmwOp::register(ctx);
    NvvmAtomicCmpxchgOp::register(ctx);
    NvvmAtomAddF16x2Op::register(ctx);
    NvvmAtomAddBf16x2Op::register(ctx);
}
