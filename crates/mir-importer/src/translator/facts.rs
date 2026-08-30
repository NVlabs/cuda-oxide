/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! The importer's rustc oracle: every question we ask rustc through the
//! `rustc_public` typed APIs is answered here.
//!
//! ```text
//! rustc (typed APIs) ──► facts.rs ──► the rest of the importer
//! ```
//!
//! The contract: a fact is either read precisely from a typed API, or it is
//! a hard error. Never a guess, never a Debug string. Identity checks
//! (is this cuda-device's `SharedArray`?) are the one soft spot: a miss
//! falls through to generic handling, it does not error.
//!
//! New rustc questions go HERE, nowhere else.
//!
//! # PointerOrigin
//!
//! Concrete pointer kinds (`&T`, `&mut T`, `*const T`, `*mut T`) are minted
//! only here. Evidence goes in, a kinded type comes out:
//!
//! ```text
//! rustc evidence                    named ABI rules
//! (Ty, BorrowKind, RawPtrKind,     (DynamicSharedArray result,
//!  Mutability, in-IR carriers)      DisjointSlice data ptr)
//!        │                                │
//!        └────────► PointerOrigin ◄───────┘
//!                        │
//!                     mint_*  ──► kinded MirPtrType / MirSliceType
//! ```
//!
//! The raw `*_with_kind` constructors are clippy-banned everywhere else in
//! the workspace, so a concrete kind can't be conjured without one of the
//! origins above. Erased pointers need no origin; use the plain
//! constructors (`MirPtrType::get*`, `MirSliceType::get*`).

use crate::error::{TranslationErr, TranslationResult};
use pliron::location::Location;
use pliron::{input_err, input_error, input_error_noloc};
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::ty::{AdtKind, ConstantKind};
use rustc_public_bridge::IndexedVal;

// ============================================================================
// Driver-provided facts (resolved with a TyCtxt the translator lacks)
// ============================================================================

/// Lang-item `DefId`s the type translator compares projections against.
///
/// The translator has no `TyCtxt`, so the driver (which does) resolves these
/// once per compilation and hands them in through `run_pipeline`:
///
/// ```text
/// driver (TyCtxt) --resolve lang items--> KnownDefs --> run_pipeline
///                                                           |
///                              translate_type <-- thread_local
/// ```
///
/// `None` fields simply never match, so a missing id degrades to the
/// existing "Alias type not yet supported" hard error instead of a guess.
#[derive(Clone, Copy, Default)]
pub struct KnownDefs {
    /// The `FnOnce::Output` associated type. `Fn` and `FnMut` declare no
    /// `Output` of their own (they inherit `FnOnce`'s), so this one id
    /// covers projections through all three traits.
    pub fn_once_output: Option<rustc_public::DefId>,
    /// The `core::ops::Index` trait.
    pub index_trait: Option<rustc_public::DefId>,
    /// The `core::ops::IndexMut` trait.
    pub index_mut_trait: Option<rustc_public::DefId>,
}

thread_local! {
    // Freshly set at every `run_pipeline` entry. The stable `DefId`s inside
    // are only meaningful within the surrounding `rustc_internal::run`
    // context, so they must never be cached beyond a single pipeline run.
    static KNOWN_DEFS: std::cell::Cell<KnownDefs> = const {
        std::cell::Cell::new(KnownDefs {
            fn_once_output: None,
            index_trait: None,
            index_mut_trait: None,
        })
    };
}

/// Installs the driver-resolved lang-item ids for this pipeline run,
/// replacing whatever a previous run on this thread left behind.
pub(crate) fn set_known_defs(defs: KnownDefs) {
    KNOWN_DEFS.with(|cell| cell.set(defs));
}

/// The lang-item ids for the current pipeline run (all `None` if the driver
/// never provided them).
pub(crate) fn known_defs() -> KnownDefs {
    KNOWN_DEFS.with(|cell| cell.get())
}

// ============================================================================
// Identity facts (crate-anchored; a miss falls through, never errors)
// ============================================================================

/// `true` when `adt_def` is cuda-device's type named `name`.
///
/// Special CUDA types (`SharedArray`, `Barrier`, ...) are recognised by bare
/// type name, so the defining-crate anchor is what keeps a user type that
/// merely shares the name out of the special-case translation. On a miss,
/// callers fall through to generic ADT handling — never an error.
pub(crate) fn is_cuda_device_adt(adt_def: &rustc_public::ty::AdtDef, name: &str) -> bool {
    adt_def.trimmed_name() == name && adt_def.krate().name.as_str() == "cuda_device"
}

/// `true` if `func` is an fn item defined in the `cuda_device` crate.
///
/// Anchors name-substring dispatch gates (e.g. `DynamicSharedArray::get`)
/// so a user fn merely spelled like a cuda-device one can't hijack an
/// intrinsic lowering; it falls through to ordinary call handling instead.
pub(crate) fn is_cuda_device_fn(func: &mir::Operand) -> bool {
    use rustc_public::ty::{RigidTy, TyKind};

    let mir::Operand::Constant(constant) = func else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::FnDef(definition, _)) = constant.const_.ty().kind() else {
        return false;
    };
    definition.krate().name.as_str() == "cuda_device"
}

/// True when `fn_def` is a `precondition_check` shim generated by libcore's
/// `assert_unsafe_precondition!` macro: a nested `const fn` literally named
/// `precondition_check` inside the guarded function (e.g.
/// `core::num::<impl usize>::unchecked_sub::precondition_check`). The macro
/// expands in `core` and (for `Vec` internals) `alloc` — the same crates
/// whose shim bodies the collector skips. Crate anchor + exact tail-segment
/// match, so a user fn such as `my_precondition_check` stays a normal call.
pub(crate) fn is_std_precondition_check(fn_def: &rustc_public::ty::FnDef) -> bool {
    if !matches!(fn_def.krate().name.as_str(), "core" | "alloc") {
        return false;
    }
    let name = fn_def.name();
    name.as_str().rsplit("::").next() == Some("precondition_check")
}

/// `true` if a trait-method call's `Self` type is `cuda_device::SharedArray`.
///
/// rustc puts a trait method's substs in declaration order, `Self` first:
///
/// ```text
/// Index::index on SharedArray<f32, 256>
///   substs = [ SharedArray<f32, 256, 0>,  usize ]
///              ^ Self at position 0       ^ Idx
/// ```
///
/// A miss is a legitimate fall-through (indexing some other type), not an
/// error. Used by both `values::classify_call` and the Index/IndexMut
/// dispatch in `translator::terminator` -- one helper, so the two can't
/// drift.
pub(crate) fn self_ty_is_shared_array(substs: &rustc_public::ty::GenericArgs) -> bool {
    use rustc_public::ty::{GenericArgKind, RigidTy, TyKind};
    let Some(GenericArgKind::Type(ty)) = substs.0.first() else {
        return false;
    };
    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = ty.kind() else {
        return false;
    };
    is_cuda_device_adt(&adt_def, "SharedArray")
}

// ============================================================================
// Pointer-origin facts (the only importer path to kinded pointer types)
// ============================================================================

// The one place in the importer allowed to touch the clippy-banned kinded
// constructors: every mint below is justified by a rustc-witnessed origin.
#[allow(clippy::disallowed_methods)]
mod pointer_origin {
    use dialect_mir::types::{MirPointerKind, MirPtrType, MirSliceType};
    use pliron::context::Context;
    use pliron::r#type::{TypeHandle, TypedHandle};
    use rustc_public::mir;

    /// A rustc-witnessed origin for a Rust pointer category.
    ///
    /// Fields are private on purpose: the only way to obtain one is to show
    /// rustc evidence to a constructor below (or to invoke a named ABI rule),
    /// so a concrete [`MirPointerKind`] can never appear out of thin air.
    #[derive(Clone, Copy, Debug)]
    pub(crate) struct PointerOrigin {
        kind: MirPointerKind,
        mutable: bool,
    }

    impl PointerOrigin {
        /// Machine-level mutability implied by the origin (carrier
        /// mutability for propagated `Erased` origins).
        pub(crate) fn is_mutable(self) -> bool {
            self.mutable
        }
    }

    // --- rustc-derived origins ---------------------------------------------

    /// `Some((pointee, origin))` iff `ty` is a reference (`RigidTy::Ref`) or
    /// a raw pointer (`RigidTy::RawPtr`).
    ///
    /// This is the signature-level coupler: every rustc-declared
    /// param/return/local/constant pointer type flows through here on its way
    /// into `dialect-mir`.
    pub(crate) fn pointer_origin_of_ty(
        ty: &rustc_public::ty::Ty,
    ) -> Option<(rustc_public::ty::Ty, PointerOrigin)> {
        use rustc_public::ty::{RigidTy, TyKind};
        match ty.kind() {
            TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
                Some((pointee, pointer_origin_of_raw_mutability(mutability)))
            }
            TyKind::RigidTy(RigidTy::Ref(_region, pointee, mutability)) => {
                let mutable = mutability == mir::Mutability::Mut;
                Some((
                    pointee,
                    PointerOrigin {
                        kind: MirPointerKind::from_reference_mutability(mutable),
                        mutable,
                    },
                ))
            }
            _ => None,
        }
    }

    /// From an `Rvalue::Ref`'s borrow kind: `Mut { .. }` is `UniqueRef`,
    /// everything else (shared and fake borrows) is `SharedRef`.
    pub(crate) fn pointer_origin_of_borrow(kind: mir::BorrowKind) -> PointerOrigin {
        let mutable = matches!(kind, mir::BorrowKind::Mut { .. });
        PointerOrigin {
            kind: MirPointerKind::from_reference_mutability(mutable),
            mutable,
        }
    }

    /// From an `Rvalue::AddressOf`'s raw-pointer kind: `Mut` is `RawMut`,
    /// `Const` and `FakeForPtrMetadata` are `RawConst`.
    pub(crate) fn pointer_origin_of_raw_ptr(kind: mir::RawPtrKind) -> PointerOrigin {
        let mutable = matches!(kind, mir::RawPtrKind::Mut);
        PointerOrigin {
            kind: MirPointerKind::from_raw_mutability(mutable),
            mutable,
        }
    }

    /// From a raw-pointer `Mutability` (e.g. an `AggregateKind::RawPtr`).
    pub(crate) fn pointer_origin_of_raw_mutability(m: mir::Mutability) -> PointerOrigin {
        let mutable = m == mir::Mutability::Mut;
        PointerOrigin {
            kind: MirPointerKind::from_raw_mutability(mutable),
            mutable,
        }
    }

    /// Propagation, never strengthening: reuse the kind and mutability an
    /// in-IR fat-pointer carrier already holds (`Erased` stays `Erased`).
    pub(crate) fn pointer_origin_of_slice_carrier(slice: &MirSliceType) -> PointerOrigin {
        PointerOrigin {
            kind: slice.pointer_kind(),
            mutable: slice.is_mutable(),
        }
    }

    /// Thin-pointer analog of [`pointer_origin_of_slice_carrier`]: address
    /// projections and address-space retypes copy the base pointer's kind
    /// verbatim.
    pub(crate) fn pointer_origin_of_ptr_carrier(ptr: &MirPtrType) -> PointerOrigin {
        PointerOrigin {
            kind: ptr.pointer_kind(),
            mutable: ptr.is_mutable(),
        }
    }

    // --- named ABI rules -----------------------------------------------------

    /// ABI RULE: `cuda_device::DynamicSharedArray`'s public API hands out
    /// `*mut T`. Extern-shared storage and its internal arithmetic stay
    /// `Erased`; this is the one boundary where the result becomes `RawMut`.
    pub(crate) fn abi_dynamic_shared_array_result() -> PointerOrigin {
        PointerOrigin {
            kind: MirPointerKind::RawMut,
            mutable: true,
        }
    }

    /// ABI RULE: `cuda_device::DisjointSlice<'_, T>` stores its data pointer
    /// as `*mut T`, so its data/element pointers are always `RawMut`.
    pub(crate) fn abi_disjoint_slice_data_ptr() -> PointerOrigin {
        PointerOrigin {
            kind: MirPointerKind::RawMut,
            mutable: true,
        }
    }

    // --- minting: the only importer path to kinded pointer/slice types ------

    /// Kinded pointer with an explicit address space.
    pub(crate) fn mint_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        address_space: u32,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_with_kind(ctx, pointee, origin.mutable, address_space, origin.kind)
    }

    /// Kinded pointer in the generic address space (0).
    pub(crate) fn mint_generic_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_generic_with_kind(ctx, pointee, origin.mutable, origin.kind)
    }

    /// Kinded pointer in the shared-memory address space (3).
    pub(crate) fn mint_shared_ptr_type(
        ctx: &mut Context,
        pointee: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirPtrType> {
        MirPtrType::get_shared_with_kind(ctx, pointee, origin.mutable, origin.kind)
    }

    /// Kinded slice/fat-pointer carrier.
    pub(crate) fn mint_slice_type(
        ctx: &mut Context,
        element_ty: TypeHandle,
        origin: PointerOrigin,
    ) -> TypedHandle<MirSliceType> {
        MirSliceType::get_with_mutability_and_kind(ctx, element_ty, origin.mutable, origin.kind)
    }
}

pub(crate) use pointer_origin::*;

// ============================================================================
// Constant facts (typed reads; exact or hard error)
// ============================================================================

/// Evaluate a const generic to a target usize. Translation runs on
/// monomorphized bodies, so a const that doesn't evaluate means
/// polymorphic MIR reached codegen: hard error, never a guess.
///
/// `what` names the const generic in the error (e.g. `"SharedArray N"`);
/// `loc` attaches a source location when the caller has one.
pub(crate) fn eval_usize_const(
    c: &rustc_public::ty::TyConst,
    what: &str,
    loc: Option<&Location>,
) -> TranslationResult<u64> {
    c.eval_target_usize().map_err(|e| {
        let err = TranslationErr::unsupported(format!(
            "{what} const generic did not evaluate to a target usize: {e:?}"
        ));
        match loc {
            Some(loc) => input_error!(loc.clone(), err),
            None => input_error_noloc!(err),
        }
    })
}

/// Read an enum constant's tag and map it to `(variant index, variant name)`.
///
/// Only valid for direct-tagged enums (e.g. `#[repr(u8)]` fieldless enums like
/// `cuda_device::atomic::AtomicOrdering`) where the constant's allocation IS
/// the tag:
///
/// ```text
/// alloc bytes [0x02] --read_uint--> tag 2 --discriminant match--> (2, "Release")
/// ```
///
/// Niche-encoded enums store no direct tag, so this mapping would be wrong for
/// them; the discriminant match errors out instead of guessing. Every failure
/// here is a hard error: inventing a variant would silently change semantics
/// (the old Debug-string scrape defaulted to variant 0, turning SeqCst atomics
/// into Relaxed ones).
pub(crate) fn extract_enum_variant(
    mir_const: &rustc_public::ty::MirConst,
    loc: &Location,
) -> TranslationResult<(usize, String)> {
    use rustc_public::ty::{RigidTy, TyConstKind, TyKind, VariantIdx};

    let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = mir_const.ty().kind() else {
        return input_err!(
            loc.clone(),
            TranslationErr::type_error(format!(
                "expected an enum constant, got a constant of type {:?}",
                mir_const.ty()
            ))
        );
    };
    if adt_def.kind() != AdtKind::Enum {
        return input_err!(
            loc.clone(),
            TranslationErr::type_error(format!(
                "expected an enum constant, got a {:?} constant of type {}",
                adt_def.kind(),
                adt_def.trimmed_name()
            ))
        );
    }

    // Pull the raw tag out of the constant. read_uint() refuses uninitialized
    // bytes, so a malformed const errors instead of yielding a made-up tag.
    let (tag, tag_width_bytes) = match mir_const.kind() {
        ConstantKind::Allocated(alloc) => {
            let tag = alloc.read_uint().map_err(|e| {
                input_error!(
                    loc.clone(),
                    TranslationErr::invalid_op(format!(
                        "cannot read the tag of enum {} from its const allocation: {e:?}",
                        adt_def.trimmed_name()
                    ))
                )
            })?;
            (tag, alloc.bytes.len())
        }
        ConstantKind::Ty(ty_const) => match ty_const.kind() {
            TyConstKind::Value(_, alloc) => {
                let tag = alloc.read_uint().map_err(|e| {
                    input_error!(
                        loc.clone(),
                        TranslationErr::invalid_op(format!(
                            "cannot read the tag of enum {} from its const allocation: {e:?}",
                            adt_def.trimmed_name()
                        ))
                    )
                })?;
                (tag, alloc.bytes.len())
            }
            other => {
                return input_err!(
                    loc.clone(),
                    TranslationErr::unsupported(format!(
                        "non-value type-level enum constant: {other:?}"
                    ))
                );
            }
        },
        ConstantKind::ZeroSized => {
            // No tag bytes to read; only unambiguous when there is exactly
            // one variant to pick.
            let variants = adt_def.variants();
            if variants.len() == 1 {
                return Ok((0, variants[0].name()));
            }
            return input_err!(
                loc.clone(),
                TranslationErr::invalid_op(format!(
                    "zero-sized constant of enum {} which has {} variants: no tag to read",
                    adt_def.trimmed_name(),
                    variants.len()
                ))
            );
        }
        ConstantKind::Unevaluated(unevaluated) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "unevaluated enum constant {:?}; use a literal variant",
                    unevaluated.def
                ))
            );
        }
        ConstantKind::Param(param) => {
            return input_err!(
                loc.clone(),
                TranslationErr::unsupported(format!(
                    "unmonomorphized const param {} used as an enum value",
                    param.name
                ))
            );
        }
    };

    if tag_width_bytes == 0 {
        return input_err!(
            loc.clone(),
            TranslationErr::invalid_op(format!(
                "empty const allocation for enum {}: no tag to read",
                adt_def.trimmed_name()
            ))
        );
    }

    // The stored tag is truncated to the physical tag width, while
    // discriminant_for_variant reports full-width values; compare masked
    // (same trick as rvalue's discriminant_to_variant_index).
    let mask = if tag_width_bytes >= 16 {
        u128::MAX
    } else {
        (1u128 << (tag_width_bytes * 8)) - 1
    };
    for (idx, variant) in adt_def.variants().iter().enumerate() {
        let discr = adt_def.discriminant_for_variant(VariantIdx::to_val(idx));
        if discr.val & mask == tag & mask {
            return Ok((idx, variant.name()));
        }
    }
    input_err!(
        loc.clone(),
        TranslationErr::invalid_op(format!(
            "enum {} has no variant with tag {tag}",
            adt_def.trimmed_name()
        ))
    )
}
