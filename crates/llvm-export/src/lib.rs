/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! LLVM dialect for cuda-oxide.
//!
//! The dialect *modeling* (types, ops, attributes, op-interfaces) now lives
//! upstream in [`pliron_llvm`]; this crate is a thin shim that re-exports it so
//! existing `llvm_export::{ops,types,attributes,op_interfaces}` paths keep
//! resolving, plus the small set of GPU-specific extensions pliron-llvm does
//! not carry (named address spaces, syncscope enum, fp16 bit helpers). The
//! pure-Rust textual `.ll` exporter ([`export`]) stays here: pliron-llvm only
//! emits real `.ll` via an `llvm-sys` bridge, which is exactly what cuda-oxide
//! is avoiding.
//!
//! Registration is automatic: every dialect/op/type/attribute linked into the
//! binary registers itself when a [`pliron::context::Context`] is created
//! (`Context::default` runs all link-time `CONTEXT_REGISTRATIONS`), so no
//! explicit `register()` entry point is needed.

pub mod export;

/// LLVM types: re-exported from pliron-llvm, plus GPU address-space helpers.
pub mod types {
    pub use pliron_llvm::types::*;

    /// `f16` maps to pliron core's builtin `FP16Type`.
    pub use pliron::builtin::types::FP16Type as HalfType;

    /// NVVM address spaces (generic=0, global=1, shared=3, constant=4,
    /// local=5, tmem=6). pliron-llvm's `PointerType` stores a raw `u32`
    /// address space with no named constants, so we keep these here.
    pub mod address_space {
        /// Generic / flat address space.
        pub const GENERIC: u32 = 0;
        /// Global memory.
        pub const GLOBAL: u32 = 1;
        /// Shared (CTA) memory.
        pub const SHARED: u32 = 3;
        /// Constant memory.
        pub const CONSTANT: u32 = 4;
        /// Thread-local memory.
        pub const LOCAL: u32 = 5;
        /// Tensor memory (Blackwell tcgen05).
        pub const TMEM: u32 = 6;
    }

    use pliron::{context::Context, r#type::TypePtr};
    pub use pliron_llvm::types::PointerType;

    /// Address-space convenience constructors/predicates re-homed from the
    /// pre-migration local `PointerType`. Upstream ships only
    /// `PointerType::get(ctx, address_space)` + `address_space()`.
    pub trait PointerTypeExt {
        /// Pointer into the generic address space.
        fn get_generic(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into the shared address space.
        fn get_shared(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into the global address space.
        fn get_global(ctx: &mut Context) -> TypePtr<PointerType>;
        /// Pointer into tensor memory.
        fn get_tmem(ctx: &mut Context) -> TypePtr<PointerType>;
        /// True if this pointer is in the shared address space.
        fn is_shared(&self) -> bool;
        /// True if this pointer is in tensor memory.
        fn is_tmem(&self) -> bool;
    }

    impl PointerTypeExt for PointerType {
        fn get_generic(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::GENERIC)
        }
        fn get_shared(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::SHARED)
        }
        fn get_global(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::GLOBAL)
        }
        fn get_tmem(ctx: &mut Context) -> TypePtr<PointerType> {
            PointerType::get(ctx, address_space::TMEM)
        }
        fn is_shared(&self) -> bool {
            self.address_space() == address_space::SHARED
        }
        fn is_tmem(&self) -> bool {
            self.address_space() == address_space::TMEM
        }
    }
}

/// LLVM attributes: re-exported from pliron-llvm, plus the syncscope enum and
/// the cuda-oxide names for atomic ordering / rmw-kind.
pub mod attributes {
    pub use pliron_llvm::attributes::*;

    /// `f16` constants use pliron core's builtin `FPHalfAttr`.
    pub use pliron::builtin::attributes::FPHalfAttr;

    /// Atomic ordering / rmw-kind were named `Llvm*` locally; upstream calls
    /// them `Atomic*Attr`. Keep the local names resolving.
    pub use pliron_llvm::attributes::{
        AtomicOrderingAttr as LlvmAtomicOrdering, AtomicRmwKindAttr as LlvmAtomicRmwKind,
    };

    /// Synchronization scope for atomics. pliron-llvm models syncscope as a
    /// free-form `Option<String>` (None = system); cuda-oxide only emits these
    /// three scopes, so we keep the enum at the lowering boundary and translate
    /// to pliron's representation via [`LlvmSyncScope::to_pliron`].
    #[derive(Clone, Copy, PartialEq, Eq, Debug)]
    pub enum LlvmSyncScope {
        /// System-wide scope (`syncscope("")` / default).
        System,
        /// Device (GPU) scope.
        Device,
        /// Block / CTA scope.
        Block,
    }

    impl LlvmSyncScope {
        /// Map to pliron's free-form syncscope string (`None` = system).
        pub fn to_pliron(self) -> Option<String> {
            match self {
                LlvmSyncScope::System => None,
                LlvmSyncScope::Device => Some("device".to_string()),
                LlvmSyncScope::Block => Some("block".to_string()),
            }
        }
    }
}

/// LLVM ops: re-exported from pliron-llvm, plus the builtin `ConstantOp` and a
/// convergent inline-asm constructor.
pub mod ops {
    pub use pliron_llvm::ops::*;

    /// `ConstantOp` moved from the LLVM dialect to pliron core `builtin`.
    pub use pliron::builtin::ops::ConstantOp;

    use pliron::{
        context::{Context, Ptr},
        identifier::Identifier,
        op::Op,
        operation::Operation,
        r#type::TypeObj,
        value::Value,
    };
    use pliron_llvm::attributes::AlignmentAttr;
    pub use pliron_llvm::ops::{GlobalOp, InlineAsmOp};

    /// Convergent inline-asm constructor re-homed from the pre-migration local
    /// op. Upstream `InlineAsmOp::new` takes a trailing `convergent: bool`;
    /// this keeps the `new_convergent(...)` call shape used across mir-lower.
    pub trait InlineAsmOpExt {
        /// Build a convergent `InlineAsmOp` (use a void result type for asm
        /// with no result value).
        fn new_convergent(
            ctx: &mut Context,
            result_ty: Ptr<TypeObj>,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
        ) -> Self;
    }

    impl InlineAsmOpExt for InlineAsmOp {
        fn new_convergent(
            ctx: &mut Context,
            result_ty: Ptr<TypeObj>,
            inputs: Vec<Value>,
            asm_template: &str,
            constraints: &str,
        ) -> Self {
            InlineAsmOp::new(ctx, result_ty, inputs, asm_template, constraints, true)
        }
    }

    /// Op-attribute key for a `GlobalOp`'s explicit alignment.
    const GLOBAL_ALIGNMENT_KEY: &str = "cuda_oxide_global_alignment";

    /// Op-attribute key under which a memory op's (`load` / `store` / `alloca`)
    /// explicit ABI alignment is stashed. Stamped by the mir-lower alignment
    /// pre-pass (while types are still MIR, so `repr(align(N))` is visible)
    /// and emitted as `align N` during export.
    const OP_ALIGNMENT_KEY: &str = "cuda_oxide_op_alignment";

    /// Stamp the ABI alignment (bytes) onto a memory op.
    pub fn set_op_alignment(ctx: &mut Context, op: Ptr<Operation>, align: u32) {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref_mut(ctx).attributes.set(key, AlignmentAttr(align));
    }

    /// Read the ABI alignment (bytes) stamped on a memory op, if any.
    pub fn op_alignment(ctx: &Context, op: Ptr<Operation>) -> Option<u32> {
        let key = Identifier::try_new(OP_ALIGNMENT_KEY.to_string()).expect("valid identifier");
        op.deref(ctx)
            .attributes
            .get::<AlignmentAttr>(&key)
            .map(|a| a.0)
    }

    /// Alignment helpers re-homed from the pre-migration local `GlobalOp`.
    /// Upstream `GlobalOp` carries type/linkage/addrspace but no alignment, so
    /// we keep the alignment in the op's generic attribute dictionary. Address
    /// space uses upstream's native `address_space` / `set_address_space`.
    pub trait GlobalOpExt {
        /// Build a `GlobalOp` carrying an explicit alignment (bytes).
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: Ptr<TypeObj>,
            alignment: u64,
        ) -> Self;
        /// Read the explicit alignment (bytes), if one was set.
        fn get_alignment(&self, ctx: &Context) -> Option<u64>;
    }

    impl GlobalOpExt for GlobalOp {
        fn new_with_alignment(
            ctx: &mut Context,
            name: Identifier,
            ty: Ptr<TypeObj>,
            alignment: u64,
        ) -> Self {
            let op = GlobalOp::new(ctx, name, ty);
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            op.get_operation()
                .deref_mut(ctx)
                .attributes
                .set(key, AlignmentAttr(alignment as u32));
            op
        }

        fn get_alignment(&self, ctx: &Context) -> Option<u64> {
            let key =
                Identifier::try_new(GLOBAL_ALIGNMENT_KEY.to_string()).expect("valid identifier");
            self.get_operation()
                .deref(ctx)
                .attributes
                .get::<AlignmentAttr>(&key)
                .map(|a| a.0 as u64)
        }
    }
}

/// LLVM op-interfaces, re-exported from pliron-llvm.
pub mod op_interfaces {
    pub use pliron_llvm::op_interfaces::*;
}

use pliron::builtin::attributes::FPHalfAttr;
use pliron::utils::apfloat::{Float, Half};

/// Build an `FPHalfAttr` from a raw 16-bit IEEE half pattern. pliron's
/// `FPHalfAttr` wraps `apfloat::Half`, whose bit access is `u128`-wide via the
/// `Float` trait, so we widen here.
pub fn fp16_attr_from_bits(bits: u16) -> FPHalfAttr {
    FPHalfAttr(Half::from_bits(bits as u128))
}

/// Extract the raw 16-bit IEEE half pattern from an `FPHalfAttr`.
pub fn fp16_attr_to_bits(attr: &FPHalfAttr) -> u16 {
    attr.0.to_bits() as u16
}
