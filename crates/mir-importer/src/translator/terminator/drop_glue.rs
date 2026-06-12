/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Conservative "is this drop glue a no-op?" analysis.
//!
//! rustc keeps a MIR `Drop` terminator for any type whose destructor
//! *might* do something. That includes types whose destructor turns out
//! to do nothing once generics are filled in. The motivating case is
//! `for x in arr` over a by-value array `[u32; 4]`: the loop iterates a
//! `core::array::IntoIter<u32, 4>`, whose destructor drops the
//! not-yet-yielded elements by calling `drop_in_place` on the "alive"
//! sub-slice of its element buffer. For `T = u32` the elements have no
//! destructor, so `drop_in_place::<[u32]>` resolves to rustc's empty
//! drop shim and the whole chain does nothing at runtime: it only
//! shuffles index values between local variables on the way to the
//! empty shim. The same holds for any element type without drop glue,
//! including plain `Copy` structs.
//!
//! cuda-oxide does not emit device-side `drop_in_place` calls, so the
//! importer normally rejects `Drop` terminators (a destructor that is
//! silently skipped would be a miscompile). This module lets it accept
//! the harmless ones: it walks the monomorphized drop-glue MIR and
//! proves that every reachable path does nothing observable. When the
//! proof succeeds, the `Drop` terminator can be lowered to a plain
//! branch. When it fails (or the analysis hits anything it does not
//! understand), the caller keeps the loud error, so a genuinely
//! effectful destructor can never be skipped by accident.
//!
//! What counts as "nothing observable":
//!
//! - Statements that exist only for analysis or storage bookkeeping
//!   (`StorageLive`, `Nop`, coverage markers, ...).
//! - Writes that stay inside the glue's own stack frame, i.e. the
//!   destination is a local variable and not a pointer dereference.
//!   Such writes die when the function returns.
//! - Calls and nested drops whose target passes this same check.
//!
//! Anything else (writes through pointers, asserts, inline asm, calls we
//! cannot resolve or whose body we cannot see) fails the proof.
//!
//! The walk follows constant `switchInt` discriminants, so branches that
//! the compiler has already folded shut (for example checks behind the
//! `UbChecks` flag, which is off for device builds) do not have to be
//! proven; only code that can actually execute does.

use rustc_public::mir;
use rustc_public::mir::mono::Instance;
use rustc_public::ty::{RigidTy, Ty, TyKind};

/// Cap on the call-chain depth the proof is willing to follow. Real
/// no-op drop glue is shallow (the `IntoIter` case above is two levels:
/// the `drop_in_place` shim, then `Drop::drop`). The cap only exists so
/// pathological inputs cannot make the compiler crawl a huge call graph.
const MAX_PROOF_DEPTH: usize = 16;

/// Returns true when dropping a value of `dropped_ty` is provably a
/// no-op, meaning the `Drop` terminator can be lowered to a plain
/// branch without skipping any observable destructor work.
///
/// `dropped_ty` must be fully monomorphized (no generic parameters
/// left), which holds for every body the importer translates.
pub(super) fn drop_glue_is_noop(dropped_ty: Ty) -> bool {
    let instance = Instance::resolve_drop_in_place(dropped_ty);
    instance_is_noop(&instance, &mut Vec::new())
}

/// Proves that calling `instance` does nothing observable.
///
/// `in_progress` holds the mangled names of instances currently being
/// proven further up the call stack. If we meet one of them again we
/// treat the cycle as harmless: a cycle cannot *introduce* an
/// observable effect, every effect would already have failed the proof
/// on some statement or terminator along the way.
fn instance_is_noop(instance: &Instance, in_progress: &mut Vec<String>) -> bool {
    // Fast path: a type with no drop glue at all resolves to an "empty"
    // drop shim that exists only to fill vtable slots. rustc_public
    // exposes that directly.
    if instance.is_empty_shim() {
        return true;
    }

    if in_progress.len() >= MAX_PROOF_DEPTH {
        return false;
    }

    let name = instance.mangled_name();
    if in_progress.contains(&name) {
        return true;
    }

    // No body means we cannot see what the call does (an intrinsic, a
    // foreign function, ...). The proof must fail.
    let Some(body) = instance.body() else {
        return false;
    };

    in_progress.push(name);
    let result = body_is_noop(&body, in_progress);
    in_progress.pop();
    result
}

/// Walks every block of `body` reachable from the entry block and
/// checks that nothing observable happens on the way to `return`.
fn body_is_noop(body: &mir::Body, in_progress: &mut Vec<String>) -> bool {
    let mut visited = vec![false; body.blocks.len()];
    let mut worklist: Vec<mir::BasicBlockIdx> = vec![0];

    while let Some(idx) = worklist.pop() {
        if std::mem::replace(&mut visited[idx], true) {
            continue;
        }
        let block = &body.blocks[idx];

        for stmt in &block.statements {
            if !statement_is_noop(&stmt.kind) {
                return false;
            }
        }

        match &block.terminator.kind {
            // Reaching `return` with only no-op work behind us is the
            // success case. `unreachable` cannot execute in a valid
            // program, so it cannot contribute an effect either.
            mir::TerminatorKind::Return | mir::TerminatorKind::Unreachable => {}

            mir::TerminatorKind::Goto { target } => worklist.push(*target),

            mir::TerminatorKind::SwitchInt { discr, targets } => {
                match const_operand_bits(discr) {
                    // Known discriminant: only the matching branch can
                    // run, so only that branch needs to be a no-op.
                    // This is what skips the dead "really drop the
                    // elements" branch in `IntoIter`'s destructor.
                    Some(value) => {
                        let target = targets
                            .branches()
                            .find(|(branch_value, _)| *branch_value == value)
                            .map(|(_, target)| target)
                            .unwrap_or_else(|| targets.otherwise());
                        worklist.push(target);
                    }
                    // Unknown discriminant: every branch must be a
                    // no-op.
                    None => worklist.extend(targets.all_targets()),
                }
            }

            // A nested drop is fine when the dropped value's own glue
            // passes this same proof.
            mir::TerminatorKind::Drop { place, target, .. } => {
                let Ok(place_ty) = place.ty(body.locals()) else {
                    return false;
                };
                let nested = Instance::resolve_drop_in_place(place_ty);
                if !instance_is_noop(&nested, in_progress) {
                    return false;
                }
                worklist.push(*target);
            }

            // A call is fine when we can resolve exactly which function
            // runs and that function passes this same proof. The
            // `drop_in_place` shim for a type with an `impl Drop` is a
            // single such call to `<T as Drop>::drop`.
            mir::TerminatorKind::Call {
                func,
                destination,
                target: Some(target),
                ..
            } => {
                if place_writes_through_pointer(destination) {
                    return false;
                }
                let Ok(func_ty) = func.ty(body.locals()) else {
                    return false;
                };
                let TyKind::RigidTy(RigidTy::FnDef(def, args)) = func_ty.kind() else {
                    // A function pointer or other indirect callee: we
                    // do not know what runs.
                    return false;
                };
                let Ok(callee) = Instance::resolve(def, &args) else {
                    return false;
                };
                if !instance_is_noop(&callee, in_progress) {
                    return false;
                }
                worklist.push(*target);
            }

            // Everything else (diverging calls, asserts, inline asm,
            // resume/abort) either has an effect or might not return.
            _ => return false,
        }
    }

    true
}

/// Returns true when executing this statement at runtime can have no
/// effect observable outside the enclosing function.
fn statement_is_noop(kind: &mir::StatementKind) -> bool {
    use mir::StatementKind;
    match kind {
        // Storage markers, analysis-only annotations, and literal
        // no-ops. None of these produce machine code with effects.
        StatementKind::StorageLive(_)
        | StatementKind::StorageDead(_)
        | StatementKind::Nop
        | StatementKind::ConstEvalCounter
        | StatementKind::Coverage(_)
        | StatementKind::PlaceMention(_)
        | StatementKind::FakeRead(..)
        | StatementKind::AscribeUserType { .. }
        | StatementKind::Retag(..) => true,

        // MIR rvalues are pure (they compute a value; only the
        // assignment's destination writes anything), so an assignment
        // is harmless as long as the destination stays inside this
        // function's own stack frame: a local, or a field of a local,
        // but never a pointer dereference.
        StatementKind::Assign(place, _) | StatementKind::SetDiscriminant { place, .. } => {
            !place_writes_through_pointer(place)
        }

        // `copy_nonoverlapping` writes through a raw pointer; never a
        // no-op.
        StatementKind::Intrinsic(_) => false,
    }
}

/// Returns true when writing to `place` writes through a pointer, i.e.
/// the write can land in memory that outlives the function. Writes to
/// plain locals (including fields of locals) vanish when the function
/// returns and are therefore unobservable.
fn place_writes_through_pointer(place: &mir::Place) -> bool {
    place
        .projection
        .iter()
        .any(|elem| matches!(elem, mir::ProjectionElem::Deref))
}

/// Extracts the raw bit value of a constant operand, e.g. the `false`
/// in `switchInt(false)`. Returns `None` for anything that is not a
/// fully evaluated scalar constant.
fn const_operand_bits(operand: &mir::Operand) -> Option<u128> {
    match operand {
        mir::Operand::Constant(const_op) => {
            let rustc_public::ty::ConstantKind::Allocated(alloc) = const_op.const_.kind() else {
                return None;
            };
            alloc.read_uint().ok()
        }
        // Runtime check flags (`UbChecks` and friends). Device code is
        // built with `-C debug-assertions=off`, so these evaluate to
        // false; the statement translator lowers them to a constant
        // false for the same reason. Folding them here lets the proof
        // skip check-only branches instead of having to prove them.
        mir::Operand::RuntimeChecks(_) => Some(0),
        _ => None,
    }
}
