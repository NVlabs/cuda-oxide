/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! GPU Debug and Profiling Operations
//!
//! This module provides operations for debugging and profiling GPU kernels:
//!
//! ```text
//! ┌─────────────────────────┬──────────────────────────────┬─────────────────────────┐
//! │ Operation               │ PTX / LLVM Intrinsic         │ Description             │
//! ├─────────────────────────┼──────────────────────────────┼─────────────────────────┤
//! │ ReadPtxSregClockOp      │ %clock / read.ptx.sreg.clock │ 32-bit clock counter    │
//! │ ReadPtxSregClock64Op    │ %clock64 / ...clock64        │ 64-bit clock counter    │
//! │ TrapOp                  │ trap / llvm.nvvm.trap        │ Abort kernel execution  │
//! │ BreakpointOp            │ brkpt / llvm.nvvm.brkpt      │ cuda-gdb breakpoint     │
//! │ VprintfOp               │ vprintf / call @vprintf      │ Formatted output        │
//! │ BlackBoxOp              │ empty asm sideeffect barrier │ `core::hint::black_box` │
//! └─────────────────────────┴──────────────────────────────┴─────────────────────────┘
//! ```

use pliron::{
    builtin::op_interfaces::{NOpdsInterface, NResultsInterface},
    builtin::types::{IntegerType, Signedness},
    common_traits::Verify,
    context::Context,
    context::Ptr,
    location::Located,
    op::Op,
    operation::Operation,
    result::Error,
    r#type::Typed,
    verify_err,
};
use pliron_derive::pliron_op;

// =============================================================================
// Clock/Timing Operations
// =============================================================================

/// Read the 32-bit GPU clock counter.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.clock` / PTX `%clock`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_clock",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClockOp;

impl ReadPtxSregClockOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregClockOp { op }
    }
}

impl Verify for ReadPtxSregClockOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(op.loc(), "nvvm.read_ptx_sreg_clock result must be integer");
            }
        };

        if int_ty.width() != 32 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_clock result must be 32-bit integer"
            );
        }
        Ok(())
    }
}

/// Read the 64-bit GPU clock counter.
///
/// Corresponds to `llvm.nvvm.read.ptx.sreg.clock64` / PTX `%clock64`.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 1 result of type `i64`
#[pliron_op(
    name = "nvvm.read_ptx_sreg_clock64",
    format,
    interfaces = [NOpdsInterface<0>, NResultsInterface<1>],
)]
pub struct ReadPtxSregClock64Op;

impl ReadPtxSregClock64Op {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        ReadPtxSregClock64Op { op }
    }
}

impl Verify for ReadPtxSregClock64Op {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let res = op.get_result(0);
        let ty = res.get_type(ctx);

        let ty_obj = ty.deref(ctx);
        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => {
                return verify_err!(
                    op.loc(),
                    "nvvm.read_ptx_sreg_clock64 result must be integer"
                );
            }
        };

        if int_ty.width() != 64 {
            return verify_err!(
                op.loc(),
                "nvvm.read_ptx_sreg_clock64 result must be 64-bit integer"
            );
        }
        Ok(())
    }
}

// =============================================================================
// Trap/Abort Operations
// =============================================================================

/// Abort kernel execution.
///
/// Corresponds to `llvm.nvvm.trap` / PTX `trap`.
/// When executed, terminates the kernel with an error.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.trap",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct TrapOp;

impl TrapOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        TrapOp { op }
    }
}

// =============================================================================
// Debugging Operations
// =============================================================================

/// Insert a cuda-gdb breakpoint.
///
/// Corresponds to `llvm.nvvm.brkpt` / PTX `brkpt`.
/// When debugging with cuda-gdb, execution stops at this point.
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.brkpt",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct BreakpointOp;

impl BreakpointOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        BreakpointOp { op }
    }
}

// =============================================================================
// Profiler Operations
// =============================================================================

/// Trigger a profiler event.
///
/// Corresponds to PTX `pmevent N;` instruction.
/// Signals the NVIDIA profiler (Nsight Systems/Compute) at this point.
///
/// The event ID is stored as an attribute (compile-time constant).
///
/// # Attributes
///
/// * `event_id` - The profiler event ID (u32)
///
/// # Verification
///
/// - Must have 0 operands
/// - Must have 0 results
#[pliron_op(
    name = "nvvm.pmevent",
    format,
    verifier = "succ",
    interfaces = [NOpdsInterface<0>, NResultsInterface<0>],
)]
pub struct PmEventOp;

impl PmEventOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        PmEventOp { op }
    }

    /// Create a new pmevent operation with the given event ID.
    pub fn new_with_event_id(ctx: &mut Context, event_id: u32) -> Ptr<Operation> {
        let op = Operation::new(ctx, Self::get_concrete_op_info(), vec![], vec![], vec![], 0);

        use pliron::builtin::attributes::IntegerAttr;
        use pliron::identifier::Identifier;
        use pliron::utils::apint::APInt;
        use std::num::NonZeroUsize;

        let i32_ty = IntegerType::get(ctx, 32, Signedness::Unsigned);
        let apint = APInt::from_u64(event_id as u64, NonZeroUsize::new(32).unwrap());
        let attr = IntegerAttr::new(i32_ty, apint);
        let key = Identifier::try_from("event_id").unwrap();
        op.deref_mut(ctx).attributes.set(key, attr);

        op
    }

    /// Get the event ID from the operation's attributes.
    pub fn get_event_id(&self, ctx: &Context) -> Option<u32> {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::identifier::Identifier;

        let key = Identifier::try_from("event_id").unwrap();
        let op_ref = self.get_operation().deref(ctx);
        let int_attr: &IntegerAttr = op_ref.attributes.get(&key)?;
        Some(int_attr.value().to_u64() as u32)
    }
}

// =============================================================================
// Printf Operations
// =============================================================================

/// GPU vprintf operation for formatted output.
///
/// Corresponds to CUDA's device-side `vprintf(format, args)` function.
/// The GPU stores format pointer and arguments to a FIFO buffer,
/// which the host reads and formats during synchronization.
///
/// # Operands
///
/// * `format_ptr` - Pointer to null-terminated format string (i8*)
/// * `args_ptr` - Pointer to packed argument buffer (i8*)
///
/// # Results
///
/// * `i32` - Number of arguments on success, negative on error
///
/// # Verification
///
/// - Must have 2 operands (format_ptr, args_ptr)
/// - Must have 1 result of type `i32`
#[pliron_op(
    name = "nvvm.vprintf",
    format,
    interfaces = [NOpdsInterface<2>, NResultsInterface<1>],
)]
pub struct VprintfOp;

impl VprintfOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        VprintfOp { op }
    }

    /// Create a new vprintf operation.
    ///
    /// # Arguments
    ///
    /// * `ctx` - The context
    /// * `format_ptr` - Pointer to format string (i8*)
    /// * `args_ptr` - Pointer to packed arguments (i8*)
    ///
    /// # Returns
    ///
    /// Operation pointer with single i32 result (arg count on success)
    pub fn build(
        ctx: &mut Context,
        format_ptr: pliron::value::Value,
        args_ptr: pliron::value::Value,
    ) -> Ptr<Operation> {
        let i32_ty = IntegerType::get(ctx, 32, Signedness::Signed);

        Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![i32_ty.to_ptr()],      // Result: i32
            vec![format_ptr, args_ptr], // Operands: format_ptr, args_ptr
            vec![],
            0,
        )
    }
}

impl Verify for VprintfOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);

        let res = op.get_result(0);
        let ty = res.get_type(ctx);
        let ty_obj = ty.deref(ctx);

        let int_ty = match ty_obj.downcast_ref::<IntegerType>() {
            Some(ty) => ty,
            None => return verify_err!(op.loc(), "nvvm.vprintf result must be integer"),
        };

        if int_ty.width() != 32 {
            return verify_err!(op.loc(), "nvvm.vprintf result must be 32-bit integer");
        }

        Ok(())
    }
}

// =============================================================================
// Compiler-Hint Operations
// =============================================================================

/// `core::hint::black_box` opaque-identity barrier.
///
/// One operand, one result of the same type. Semantically the identity
/// function — at the LLVM stage this lowers to an empty inline `asm
/// sideeffect` with register input/output (the same shape rustc's LLVM
/// backend emits), which LLVM's optimizer treats as opaque. This is
/// what prevents `black_box(const)` from being const-folded back into
/// a constant, which is the whole point of the intrinsic.
///
/// Lives in `dialect-nvvm` rather than `dialect-mir` because the
/// only lowering we know how to express today is target-specific
/// (PTX inline asm constraint letters).
///
/// # Verification
///
/// - Must have 1 operand
/// - Must have 1 result of the same type as the operand
#[pliron_op(
    name = "nvvm.black_box",
    format,
    interfaces = [NOpdsInterface<1>, NResultsInterface<1>],
)]
pub struct BlackBoxOp;

impl BlackBoxOp {
    /// Wrap an existing operation pointer.
    pub fn new(op: Ptr<Operation>) -> Self {
        BlackBoxOp { op }
    }
}

impl Verify for BlackBoxOp {
    fn verify(&self, ctx: &Context) -> Result<(), Error> {
        let op = &*self.get_operation().deref(ctx);
        let in_ty = op.get_operand(0).get_type(ctx);
        let out_ty = op.get_result(0).get_type(ctx);
        if in_ty != out_ty {
            return verify_err!(
                op.loc(),
                "nvvm.black_box operand and result must have the same type"
            );
        }
        Ok(())
    }
}

/// Register debug operations with the context.
pub(super) fn register(ctx: &mut Context) {
    // Clock/Timing
    ReadPtxSregClockOp::register(ctx);
    ReadPtxSregClock64Op::register(ctx);
    // Trap
    TrapOp::register(ctx);
    // Debugging
    BreakpointOp::register(ctx);
    // Profiler
    PmEventOp::register(ctx);
    // Printf
    VprintfOp::register(ctx);
    // Compiler hints
    BlackBoxOp::register(ctx);
}
