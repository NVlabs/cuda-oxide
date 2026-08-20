/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Function body translation: MIR → `mir.func`.
//!
//! Translates complete MIR function bodies into `dialect-mir` `mir.func` operations.
//!
//! # Responsibilities
//!
//! - Extract function signature (arguments, return type)
//! - Create and link pliron IR basic blocks (entry block carries function
//!   parameters; every other block is argument-less)
//! - Emit one `mir.alloca` per non-ZST MIR local at the top of the entry
//!   block and record the slot in [`ValueMap`]
//! - Translate every reachable block in order; unwind-only cleanup blocks
//!   are patched with `mir.unreachable`
//! - Detect compile-time kernel attributes (`#[cluster(...)]`,
//!   `#[launch_bounds(...)]`)

use super::block;
use super::rvalue;
use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::location::span_to_location;
use crate::translator::values::{self, SlotAddrSpaceMap, ValueMap};
use dialect_mir::attributes::FieldIndexAttr;
use dialect_mir::ops::{MirAllocaOp, MirDbgValueOp, MirFieldAddrOp, MirFuncOp, MirStoreOp};
use dialect_mir::types::{MirPtrType, address_space};
use llvm_export::export::DebugKind;
use llvm_export::ops::{
    DebugEnumDiscriminant, DebugEnumVariant, DebugFragment, DebugFragmentVariableInfo,
    DebugLocalTypeKind, DebugLocalVariableInfo, DebugProjectedVariableInfo, DebugSourcePosition,
    DebugSourceScopeMap, DebugTypeMember, LocalMemoryProvenanceAttr,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::identifier::{Identifier, Legaliser};
use pliron::input_err_noloc;
use pliron::linked_list::ContainsLinkedList;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;
use pliron::r#type::TypeHandle;
use pliron::value::Value;

// Re-export rustc_public types for convenience
use rustc_hash::FxHashMap;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::mono;
use rustc_public::ty::{ConstantKind, FloatTy, IntTy, RigidTy, Ty, TyKind, UintTy};
use rustc_public_bridge::IndexedVal;

/// Cluster dimensions extracted from `#[cluster(x,y,z)]` attribute.
///
/// These are detected by scanning MIR for `cuda_device::cluster::__cluster_config::<X,Y,Z>()`
/// marker calls injected by the `#[cluster]` macro.
#[derive(Debug, Clone, Copy)]
pub struct ClusterDims {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Launch bounds extracted from `#[launch_bounds(max, min)]` attribute.
///
/// These are detected by scanning MIR for `cuda_device::thread::__launch_bounds_config::<MAX,MIN>()`
/// marker calls injected by the `#[launch_bounds]` macro.
#[derive(Debug, Clone, Copy)]
pub struct LaunchBounds {
    /// Maximum threads per block (.maxntid in PTX)
    pub max_threads: u32,
    /// Minimum blocks per SM (.minnctapersm in PTX), 0 if unspecified
    pub min_blocks: u32,
}

/// Exact block shape declared by `#[launch_contract(block = (x, y, z))]`.
///
/// Detected by scanning MIR for
/// `cuda_device::thread::__launch_contract_block_config::<X, Y, Z>()` marker
/// calls. Emitted as `.reqntid x, y, z`, which the CUDA driver enforces per
/// axis at launch.
#[derive(Debug, Clone, Copy)]
pub struct ContractBlock {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

/// Minimum extern-shared alignment declared by `#[launch_contract]`.
#[derive(Debug, Clone, Copy)]
pub struct DynamicSharedAlignment {
    pub bytes: u64,
}

/// Scans MIR for `__cluster_config::<X, Y, Z>()` marker and extracts cluster dimensions.
///
/// The `#[cluster(x,y,z)]` macro injects this call at the start of the kernel.
/// We scan the MIR to find it and extract the const generic parameters.
///
/// Returns `Some(ClusterDims)` if found, `None` otherwise.
fn detect_cluster_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<ClusterDims> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        // Use let-else for early continue pattern
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let fn_name = def_id.name();
        if fn_name != "__cluster_config" && !fn_name.ends_with("::__cluster_config") {
            continue;
        }

        // Extract const generic args (X, Y, Z)
        let mut dims = [1u32, 1u32, 1u32];
        for (i, arg) in args.0.iter().take(3).enumerate() {
            let rustc_public::ty::GenericArgKind::Const(c) = arg else {
                continue;
            };
            dims[i] = match c.kind() {
                TyConstKind::Value(_, alloc) => alloc.read_uint().ok().map(|v| v as u32),
                _ => c.eval_target_usize().ok().map(|v| v as u32),
            }
            .unwrap_or(dims[i]);
        }

        return Some(ClusterDims {
            x: dims[0],
            y: dims[1],
            z: dims[2],
        });
    }
    None
}

/// Scans MIR for `__launch_bounds_config::<MAX, MIN>()` marker and extracts launch bounds.
///
/// The `#[launch_bounds(max, min)]` macro injects this call at the start of the kernel.
/// We scan the MIR to find it and extract the const generic parameters.
///
/// Returns `Some(LaunchBounds)` if found, `None` otherwise.
fn detect_launch_bounds_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<Option<LaunchBounds>, String> {
    use rustc_public::ty::TyConstKind;

    let mut detected: Option<LaunchBounds> = None;
    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__launch_bounds_config"
                && !definition_name.ends_with("::__launch_bounds_config"))
        {
            continue;
        }

        if args.0.len() != 2 {
            return Err(format!(
                "cuda_device launch-bounds marker has {} generic arguments; expected exactly 2",
                args.0.len()
            ));
        }
        let mut values = [0u32; 2];
        for (index, (name, arg)) in ["maximum threads", "minimum blocks"]
            .into_iter()
            .zip(args.0.iter())
            .enumerate()
        {
            let rustc_public::ty::GenericArgKind::Const(value) = arg else {
                return Err(format!(
                    "cuda_device launch-bounds {name} argument is not a constant"
                ));
            };
            let raw = match value.kind() {
                TyConstKind::Value(_, allocation) => allocation.read_uint().map_err(|error| {
                    format!("could not read launch-bounds {name} constant: {error:?}")
                })?,
                _ => u128::from(value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate launch-bounds {name} constant: {error:?}")
                })?),
            };
            values[index] = u32::try_from(raw)
                .map_err(|_| format!("launch-bounds {name} value {raw} does not fit in u32"))?;
        }
        if values[0] == 0 {
            return Err("launch-bounds maximum threads must be greater than zero".to_string());
        }
        let bounds = LaunchBounds {
            max_threads: values[0],
            min_blocks: values[1],
        };
        if let Some(existing) = detected {
            if existing.max_threads != bounds.max_threads
                || existing.min_blocks != bounds.min_blocks
            {
                return Err(format!(
                    "a kernel contains conflicting cuda_device launch-bounds markers: ({}, {}) and ({}, {})",
                    existing.max_threads,
                    existing.min_blocks,
                    bounds.max_threads,
                    bounds.min_blocks,
                ));
            }
        } else {
            detected = Some(bounds);
        }
    }
    Ok(detected)
}

/// Scans MIR for `__launch_contract_block_config::<X, Y, Z>()` and extracts the
/// exact block shape declared by `#[launch_contract(block = (x, y, z))]`.
///
/// Returns `Some(ContractBlock)` if found, `None` otherwise.
fn detect_contract_block_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<Option<ContractBlock>, String> {
    use rustc_public::ty::TyConstKind;

    let mut detected: Option<ContractBlock> = None;
    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__launch_contract_block_config"
                && !definition_name.ends_with("::__launch_contract_block_config"))
        {
            continue;
        }

        if args.0.len() != 3 {
            return Err(format!(
                "cuda_device launch-contract block marker has {} generic arguments; expected exactly 3",
                args.0.len()
            ));
        }
        let mut values = [0u32; 3];
        for (index, (axis, arg)) in ["x", "y", "z"].into_iter().zip(args.0.iter()).enumerate() {
            let rustc_public::ty::GenericArgKind::Const(value) = arg else {
                return Err(format!(
                    "cuda_device launch-contract block {axis} argument is not a constant"
                ));
            };
            let raw = match value.kind() {
                TyConstKind::Value(_, allocation) => allocation.read_uint().map_err(|error| {
                    format!("could not read launch-contract block {axis} constant: {error:?}")
                })?,
                _ => u128::from(value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate launch-contract block {axis} constant: {error:?}")
                })?),
            };
            values[index] = u32::try_from(raw).map_err(|_| {
                format!("launch-contract block {axis} value {raw} does not fit in u32")
            })?;
            if values[index] == 0 {
                return Err(format!(
                    "launch-contract block {axis} dimension must be greater than zero"
                ));
            }
        }
        let shape = ContractBlock {
            x: values[0],
            y: values[1],
            z: values[2],
        };
        if let Some(existing) = detected {
            if existing.x != shape.x || existing.y != shape.y || existing.z != shape.z {
                return Err(format!(
                    "a kernel contains conflicting cuda_device launch-contract block markers: ({}, {}, {}) and ({}, {}, {})",
                    existing.x, existing.y, existing.z, shape.x, shape.y, shape.z,
                ));
            }
        } else {
            detected = Some(shape);
        }
    }
    Ok(detected)
}

/// Rejects an exact block shape that needs more threads than
/// `#[launch_bounds]` allows.
///
/// An exact block displaces the thread maximum in the emitted PTX, because
/// ptxas rejects an entry carrying both `.maxntid` and `.reqntid`. A maximum
/// below the required thread count would therefore be dropped in silence, and
/// the kernel would launch at a shape its author ruled out. A maximum at or
/// above the required count is redundant rather than contradictory, since
/// `.reqntid` is the stronger statement, so it stays allowed.
fn validate_block_against_bounds(bounds: LaunchBounds, block: ContractBlock) -> Result<(), String> {
    let required = u64::from(block.x) * u64::from(block.y) * u64::from(block.z);
    if required > u64::from(bounds.max_threads) {
        return Err(format!(
            "a kernel declares #[launch_contract(block = ({}, {}, {}))], needing {} threads per block, and #[launch_bounds({})], allowing at most {}",
            block.x, block.y, block.z, required, bounds.max_threads, bounds.max_threads,
        ));
    }
    Ok(())
}

/// Scans MIR for the `__unchecked_indexing_config::<ENABLED>()` marker
/// injected by `#[kernel(unchecked_indexing)]` and extracts its const bool.
///
/// Returns `Ok(true)` when a marker with `ENABLED = true` is reachable in
/// this body. The marker call itself is stripped later during terminator
/// translation; this scan only records the policy.
fn detect_unchecked_indexing_config(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Result<bool, String> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };

        let definition_name = def_id.name();
        if def_id.krate().name.as_str() != "cuda_device"
            || (definition_name != "__unchecked_indexing_config"
                && !definition_name.ends_with("::__unchecked_indexing_config"))
        {
            continue;
        }

        if args.0.len() != 1 {
            return Err(format!(
                "cuda_device unchecked-indexing marker has {} generic arguments; expected exactly 1",
                args.0.len()
            ));
        }
        let rustc_public::ty::GenericArgKind::Const(value) = &args.0[0] else {
            return Err(
                "cuda_device unchecked-indexing marker argument is not a constant".to_string(),
            );
        };
        let enabled = match value.kind() {
            TyConstKind::Value(_, allocation) => allocation.read_bool().map_err(|error| {
                format!("could not read unchecked-indexing marker constant: {error:?}")
            })?,
            _ => {
                value.eval_target_usize().map_err(|error| {
                    format!("could not evaluate unchecked-indexing marker constant: {error:?}")
                })? != 0
            }
        };
        if enabled {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the whole-build unchecked-indexing switch is on.
///
/// `CUDA_OXIDE_UNCHECKED_INDEXING=1` (or `true`) elides bounds-check asserts
/// in every translated body, including separately translated `#[device]`
/// functions that the per-kernel marker cannot reach.
fn unchecked_indexing_env_enabled() -> bool {
    std::env::var("CUDA_OXIDE_UNCHECKED_INDEXING")
        .is_ok_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

/// Scans MIR for the dynamic-shared alignment marker injected by
/// `#[launch_contract]` and extracts its const generic argument. The importer
/// records the value before removing the call from the executable path.
fn detect_dynamic_shared_alignment(
    body: &mir::Body,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<DynamicSharedAlignment> {
    use rustc_public::ty::TyConstKind;

    for &block_idx in reachable {
        let block = &body.blocks[block_idx];
        let mir::TerminatorKind::Call { func, .. } = &block.terminator.kind else {
            continue;
        };
        let mir::Operand::Constant(constant) = func else {
            continue;
        };
        let ConstantKind::ZeroSized = constant.const_.kind() else {
            continue;
        };
        let TyKind::RigidTy(RigidTy::FnDef(def_id, args)) = constant.const_.ty().kind() else {
            continue;
        };
        let fn_name = def_id.name();
        if fn_name != "__dynamic_shared_alignment"
            && !fn_name.ends_with("::__dynamic_shared_alignment")
        {
            continue;
        }
        let rustc_public::ty::GenericArgKind::Const(alignment) = args.0.first()? else {
            continue;
        };
        let bytes = match alignment.kind() {
            TyConstKind::Value(_, alloc) => alloc.read_uint().ok().map(|value| value as u64),
            _ => alignment.eval_target_usize().ok(),
        }?;
        return Some(DynamicSharedAlignment { bytes });
    }
    None
}

/// Return the non-unwind successors of a block's terminator.
///
/// [`mir::Terminator::successors`] includes unwind cleanup blocks alongside
/// "normal" control-flow targets. The CUDA toolchain does not support stack
/// unwinding (hardware could, but `nvcc`/`ptxas` never wire it up), so the
/// translator treats unwind cleanups as dead code. This helper strips them
/// out so the worklist only visits blocks that matter on GPU. Monomorphized
/// branch reachability is supplied separately by rustc's collector; the
/// importer must not reconstruct a second constant-evaluation model from the
/// converted public MIR.
fn non_unwind_successors(block: &mir::BasicBlock) -> Vec<usize> {
    use mir::TerminatorKind::*;
    match &block.terminator.kind {
        Goto { target } => vec![*target],
        SwitchInt { targets, .. } => targets.all_targets(),
        Return | Resume | Abort | Unreachable => vec![],
        Drop { target, .. } | Assert { target, .. } => vec![*target],
        Call { target, .. } => target.map(|t| vec![t]).unwrap_or_default(),
        InlineAsm { destination, .. } => destination.map(|t| vec![t]).unwrap_or_default(),
    }
}

fn validate_monomorphized_successor_shape(
    body_block_count: usize,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
) -> Result<(), String> {
    if body_block_count != rustc_mir_block_count {
        return Err(format!(
            "rustc/public MIR CFG mismatch: collector recorded {rustc_mir_block_count} blocks but importer received {body_block_count}"
        ));
    }
    if rustc_mono_successors.len() != body_block_count {
        return Err(format!(
            "rustc collector supplied successor lists for {} blocks but the public MIR body has {body_block_count}",
            rustc_mono_successors.len()
        ));
    }
    for (source, successors) in rustc_mono_successors.iter().enumerate() {
        if let Some(target) = successors
            .iter()
            .copied()
            .find(|target| *target >= body_block_count)
        {
            return Err(format!(
                "rustc collector edge {source} -> {target} is outside the {body_block_count}-block public MIR body"
            ));
        }
    }
    Ok(())
}

fn validate_monomorphized_successors(
    body: &mir::Body,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
) -> Result<(), String> {
    validate_monomorphized_successor_shape(
        body.blocks.len(),
        rustc_mir_block_count,
        rustc_mono_successors,
    )?;
    for (source, successors) in rustc_mono_successors.iter().enumerate() {
        let public_successors = body.blocks[source].terminator.successors();
        if let Some(target) = successors
            .iter()
            .copied()
            .find(|target| !public_successors.contains(target))
        {
            return Err(format!(
                "rustc collector edge {source} -> {target} does not exist in the converted public MIR CFG"
            ));
        }
    }
    Ok(())
}

/// BFS from the entry block following rustc's exact per-block monomorphized
/// successors, intersected with the importer's existing non-unwind policy.
///
/// The result is a sorted set of reachable-on-GPU block indices; unwind-only
/// cleanup blocks end up outside this set and are filled in with
/// `mir.unreachable` by [`translate_body`] so pliron verification still
/// passes. Constant switches and device runtime-check switches are never
/// re-evaluated here: the collector's edges are the semantic source of truth.
fn compute_reachable_blocks(
    body: &mir::Body,
    rustc_mono_successors: &[Vec<usize>],
) -> std::collections::BTreeSet<usize> {
    let mut reachable = std::collections::BTreeSet::new();
    let mut frontier: Vec<usize> = vec![0];
    reachable.insert(0);
    while let Some(idx) = frontier.pop() {
        let non_unwind: std::collections::BTreeSet<_> = non_unwind_successors(&body.blocks[idx])
            .into_iter()
            .collect();
        for &succ in &rustc_mono_successors[idx] {
            if non_unwind.contains(&succ) && reachable.insert(succ) {
                frontier.push(succ);
            }
        }
    }
    reachable
}

#[derive(Clone)]
struct LocalDebugInfo {
    variable: DebugLocalVariableInfo,
    loc: pliron::location::Location,
    source_scope: u32,
}

#[derive(Clone)]
struct ConstantDebugFragment {
    constant: mir::ConstOperand,
    fragment: DebugFragmentVariableInfo,
    loc: pliron::location::Location,
}

#[derive(Clone)]
enum CompositeMirrorFragmentValue {
    Constant {
        constant: mir::ConstOperand,
        loc: pliron::location::Location,
    },
    Place {
        local: mir::Local,
    },
}

#[derive(Clone)]
struct CompositeMirrorFragment {
    fragment: DebugFragmentVariableInfo,
    field_index: usize,
    field_ty: Ty,
    value: CompositeMirrorFragmentValue,
}

#[derive(Clone)]
struct CompositeDebugMirror {
    variable: DebugLocalVariableInfo,
    source_ty: Ty,
    source_scope: u32,
    declaration: Option<DebugSourcePosition>,
    loc: pliron::location::Location,
    fragments: Vec<CompositeMirrorFragment>,
}

#[derive(Clone, Copy)]
struct CompositeMirrorPlaceBinding {
    backing_slot: Value,
    mirror_slot: Value,
    field_index: usize,
    field_ty: TypeHandle,
}

#[derive(Default)]
struct CollectedDebugLocals {
    whole: FxHashMap<mir::Local, LocalDebugInfo>,
    projected: FxHashMap<mir::Local, Vec<DebugProjectedVariableInfo>>,
    fragments: FxHashMap<mir::Local, Vec<DebugFragmentVariableInfo>>,
    constant_fragments: Vec<ConstantDebugFragment>,
    composite_mirrors: Vec<CompositeDebugMirror>,
}

/// Build full-debug bindings for whole locals, supported place projections,
/// and rustc scalar-replacement fragments.
///
/// A composite record describes one storage piece of a larger source variable.
/// The stable MIR `composite.ty` is the complete source type and
/// `composite.projection` identifies the piece inside it. For now fragments are
/// accepted when rustc stores the piece in a whole MIR local or records it as
/// a constant after propagation, and the composite projection is a static
/// `Field` chain. These are the scalar-replaced aggregate shapes emitted by
/// rustc and keep the location semantics exact.
///
/// Ordinary projected bindings retain the existing support for static fields,
/// forward constant indices, enum payload fields, and one leading thin-pointer
/// dereference. Dynamic indices, dereference-index chains, repeated dereferences,
/// fat pointers, slices, opaque casts, and non-field composite projections are
/// skipped rather than approximated.
fn collect_debug_locals(ctx: &mut Context, body: &mir::Body) -> CollectedDebugLocals {
    let mut collected = CollectedDebugLocals::default();
    let mut mirror_candidates: Vec<CompositeDebugMirror> = Vec::new();
    let mut blocked_mirrors: Vec<(String, u32, Ty)> = Vec::new();

    for info in &body.var_debug_info {
        let name = info.name.to_string();
        if name.is_empty() {
            continue;
        }

        if let Some(composite) = &info.composite {
            let Some(fragment) = debug_fragment(composite) else {
                continue;
            };
            let Some(ty) = debug_type_for_ty(&composite.ty) else {
                continue;
            };
            let fragment_info = DebugFragmentVariableInfo {
                variable: DebugLocalVariableInfo {
                    name: name.clone(),
                    argument_index: info.argument_index,
                    ty,
                },
                fragment,
                source_scope: Some(info.source_info.scope),
                declaration: debug_source_position(info.source_info.span),
            };

            let direct_field = match composite.projection.as_slice() {
                [mir::ProjectionElem::Field(field_idx, field_ty)] => Some((*field_idx, *field_ty)),
                _ => None,
            };

            let mirror_key_matches = |mirror: &CompositeDebugMirror| {
                mirror.variable.name == name
                    && mirror.source_scope == info.source_info.scope
                    && mirror.source_ty == composite.ty
            };
            let blocked_key_matches = |key: &(String, u32, Ty)| {
                key.0 == name && key.1 == info.source_info.scope && key.2 == composite.ty
            };

            match &info.value {
                mir::VarDebugInfoContents::Const(constant) => {
                    if layout_size_bits(&constant.ty()) != Some(fragment.size_bits) {
                        if !blocked_mirrors.iter().any(blocked_key_matches) {
                            blocked_mirrors.push((
                                name.clone(),
                                info.source_info.scope,
                                composite.ty,
                            ));
                        }
                        continue;
                    }

                    if let Some((field_index, field_ty)) = direct_field {
                        let candidate_index = mirror_candidates
                            .iter()
                            .position(mirror_key_matches)
                            .unwrap_or_else(|| {
                                mirror_candidates.push(CompositeDebugMirror {
                                    variable: fragment_info.variable.clone(),
                                    source_ty: composite.ty,
                                    source_scope: info.source_info.scope,
                                    declaration: fragment_info.declaration.clone(),
                                    loc: span_to_location(ctx, info.source_info.span),
                                    fragments: Vec::new(),
                                });
                                mirror_candidates.len() - 1
                            });
                        mirror_candidates[candidate_index].fragments.push(
                            CompositeMirrorFragment {
                                fragment: fragment_info,
                                field_index,
                                field_ty,
                                value: CompositeMirrorFragmentValue::Constant {
                                    constant: constant.clone(),
                                    loc: span_to_location(ctx, info.source_info.span),
                                },
                            },
                        );
                        continue;
                    }

                    if !blocked_mirrors.iter().any(blocked_key_matches) {
                        blocked_mirrors.push((name.clone(), info.source_info.scope, composite.ty));
                    }
                    collected.constant_fragments.push(ConstantDebugFragment {
                        constant: constant.clone(),
                        fragment: fragment_info,
                        loc: span_to_location(ctx, info.source_info.span),
                    });
                    continue;
                }
                mir::VarDebugInfoContents::Place(place) => {
                    let local = place.local;
                    let local_idx: usize = local;
                    if local_idx == 0 {
                        if !blocked_mirrors.iter().any(blocked_key_matches) {
                            blocked_mirrors.push((
                                name.clone(),
                                info.source_info.scope,
                                composite.ty,
                            ));
                        }
                        continue;
                    }

                    // A promoted `dbg.value` for the backing local denotes the
                    // fragment value itself. Supporting a projected storage
                    // place would require extracting that subvalue after
                    // promotion, so fail closed here.
                    if !place.projection.is_empty() {
                        if !blocked_mirrors.iter().any(blocked_key_matches) {
                            blocked_mirrors.push((
                                name.clone(),
                                info.source_info.scope,
                                composite.ty,
                            ));
                        }
                        continue;
                    }

                    if fragment.offset_bits == 0
                        && layout_size_bits(&composite.ty) == Some(fragment.size_bits)
                    {
                        if !blocked_mirrors.iter().any(blocked_key_matches) {
                            blocked_mirrors.push((
                                name.clone(),
                                info.source_info.scope,
                                composite.ty,
                            ));
                        }
                        collected
                            .whole
                            .entry(local)
                            .or_insert_with(|| LocalDebugInfo {
                                variable: fragment_info.variable,
                                loc: span_to_location(ctx, info.source_info.span),
                                source_scope: info.source_info.scope,
                            });
                        continue;
                    }

                    if let Some((field_index, field_ty)) = direct_field {
                        let candidate_index = mirror_candidates
                            .iter()
                            .position(mirror_key_matches)
                            .unwrap_or_else(|| {
                                mirror_candidates.push(CompositeDebugMirror {
                                    variable: fragment_info.variable.clone(),
                                    source_ty: composite.ty,
                                    source_scope: info.source_info.scope,
                                    declaration: fragment_info.declaration.clone(),
                                    loc: span_to_location(ctx, info.source_info.span),
                                    fragments: Vec::new(),
                                });
                                mirror_candidates.len() - 1
                            });
                        mirror_candidates[candidate_index].fragments.push(
                            CompositeMirrorFragment {
                                fragment: fragment_info,
                                field_index,
                                field_ty,
                                value: CompositeMirrorFragmentValue::Place { local },
                            },
                        );
                        continue;
                    }

                    if !blocked_mirrors.iter().any(blocked_key_matches) {
                        blocked_mirrors.push((name.clone(), info.source_info.scope, composite.ty));
                    }
                    collected
                        .fragments
                        .entry(local)
                        .or_default()
                        .push(fragment_info);
                    continue;
                }
            }
        }

        let mir::VarDebugInfoContents::Place(place) = &info.value else {
            continue;
        };
        let local = place.local;
        let local_idx: usize = local;
        if local_idx == 0 {
            continue;
        }

        if place.projection.is_empty() {
            let Some(decl) = body.local_decl(local) else {
                continue;
            };
            let Some(ty) = debug_type_for_ty(&decl.ty) else {
                continue;
            };

            collected
                .whole
                .entry(local)
                .or_insert_with(|| LocalDebugInfo {
                    variable: DebugLocalVariableInfo {
                        name,
                        argument_index: info.argument_index,
                        ty,
                    },
                    loc: span_to_location(ctx, info.source_info.span),
                    source_scope: info.source_info.scope,
                });
            continue;
        }

        let Some(projection) = debug_projection(body, place) else {
            continue;
        };
        let Some(ty) = debug_type_for_ty(&projection.ty) else {
            continue;
        };

        // rustc treats projected argument bindings as local variables rather
        // than formal argument variables; only whole-place arguments receive
        // a DWARF argument index.
        collected
            .projected
            .entry(local)
            .or_default()
            .push(DebugProjectedVariableInfo {
                variable: DebugLocalVariableInfo {
                    name,
                    argument_index: None,
                    ty,
                },
                dereference_base: projection.dereference_base,
                offset_bytes: projection.offset_bytes,
                source_scope: Some(info.source_info.scope),
                declaration: debug_source_position(info.source_info.span),
            });
    }

    for mirror in mirror_candidates {
        let blocked = blocked_mirrors.iter().any(|key| {
            key.0 == mirror.variable.name
                && key.1 == mirror.source_scope
                && key.2 == mirror.source_ty
        });
        let has_constant = mirror.fragments.iter().any(|fragment| {
            matches!(
                &fragment.value,
                CompositeMirrorFragmentValue::Constant { .. }
            )
        });

        if !blocked && has_constant {
            collected.composite_mirrors.push(mirror);
            continue;
        }

        for fragment in mirror.fragments {
            match fragment.value {
                CompositeMirrorFragmentValue::Constant { constant, loc } => {
                    collected.constant_fragments.push(ConstantDebugFragment {
                        constant,
                        fragment: fragment.fragment,
                        loc,
                    });
                }
                CompositeMirrorFragmentValue::Place { local } => {
                    collected
                        .fragments
                        .entry(local)
                        .or_default()
                        .push(fragment.fragment);
                }
            }
        }
    }

    collected
}

fn debug_fragment(fragment: &mir::VarDebugInfoFragment) -> Option<DebugFragment> {
    let whole_size_bits = layout_size_bits(&fragment.ty)?;
    if whole_size_bits == 0 {
        return None;
    }

    let mut current_ty = fragment.ty;
    let mut offset_bytes = 0u64;
    for elem in &fragment.projection {
        let mir::ProjectionElem::Field(field_idx, field_ty) = elem else {
            return None;
        };
        let layout = current_ty.layout().ok()?;
        let shape = layout.shape();
        let rustc_public::abi::FieldsShape::Arbitrary { offsets } = &shape.fields else {
            return None;
        };
        offset_bytes = offset_bytes.checked_add(offsets.get(*field_idx)?.bytes() as u64)?;
        current_ty = *field_ty;
    }

    let offset_bits = offset_bytes.checked_mul(8)?;
    let size_bits = layout_size_bits(&current_ty)?;
    if size_bits == 0 || offset_bits.checked_add(size_bits)? > whole_size_bits {
        return None;
    }

    Some(DebugFragment {
        offset_bits,
        size_bits,
    })
}

#[derive(Clone, Copy)]
struct ResolvedDebugProjection {
    dereference_base: bool,
    offset_bytes: u64,
    ty: Ty,
}

/// Resolve the location expression and final type of a supported MIR projection.
///
/// Static `Field`/forward-`ConstantIndex` chains retain the #939 behavior. A
/// single leading `Deref` is additionally accepted when the base has one pointer
/// word of storage; after that dereference only `Field` projections are allowed.
/// This deliberately rejects fat references/raw pointers, repeated dereferences,
/// and dereference-plus-index chains instead of emitting an approximate location.
fn debug_projection(body: &mir::Body, place: &mir::Place) -> Option<ResolvedDebugProjection> {
    let mut current_ty = body.local_decl(place.local)?.ty;
    let mut dereference_base = false;
    let mut offset_bytes = 0u64;
    let mut enum_variant = None;

    for (index, elem) in place.projection.iter().enumerate() {
        match elem {
            mir::ProjectionElem::Deref if index == 0 && !dereference_base => {
                // CUDA device pointers are one 64-bit word. Requiring the source
                // pointer/reference layout to match rejects fat pointers such as
                // `&[T]`/`&str` before we model them with the wrong DWARF stack op.
                if current_ty.layout().ok()?.shape().size.bytes() != 8 {
                    return None;
                }
                current_ty = match current_ty.kind() {
                    TyKind::RigidTy(RigidTy::RawPtr(pointee, _)) => pointee,
                    TyKind::RigidTy(RigidTy::Ref(_, pointee, _)) => pointee,
                    _ => return None,
                };
                dereference_base = true;
            }
            mir::ProjectionElem::Downcast(variant) => {
                // Downcasts are supported only inside the base local: after a
                // dereference the tested recipe allows static fields alone.
                if enum_variant.is_some() || dereference_base {
                    return None;
                }
                let TyKind::RigidTy(RigidTy::Adt(adt_def, _)) = current_ty.kind() else {
                    return None;
                };
                if !matches!(adt_def.kind(), rustc_public::ty::AdtKind::Enum) {
                    return None;
                }
                enum_variant = Some(variant.to_index());
            }
            mir::ProjectionElem::Field(field_idx, field_ty) => {
                let layout = current_ty.layout().ok()?;
                let shape = layout.shape();
                let field_offset = if let Some(variant) = enum_variant.take() {
                    crate::translator::layout::enum_variant_field_offsets(
                        &shape,
                        variant,
                        pliron::location::Location::Unknown,
                    )
                    .ok()?
                    .get(*field_idx)
                    .copied()? as u64
                } else {
                    let rustc_public::abi::FieldsShape::Arbitrary { offsets } = &shape.fields
                    else {
                        return None;
                    };
                    offsets.get(*field_idx)?.bytes() as u64
                };
                offset_bytes = offset_bytes.checked_add(field_offset)?;
                current_ty = *field_ty;
            }
            mir::ProjectionElem::ConstantIndex {
                offset,
                min_length: _,
                from_end: false,
            } if !dereference_base => {
                if enum_variant.is_some() {
                    return None;
                }
                let layout = current_ty.layout().ok()?;
                let shape = layout.shape();
                let rustc_public::abi::FieldsShape::Array { stride, count } = &shape.fields else {
                    return None;
                };
                if *offset >= *count {
                    return None;
                }
                let element_offset = (stride.bytes() as u64).checked_mul(*offset)?;
                offset_bytes = offset_bytes.checked_add(element_offset)?;
                let TyKind::RigidTy(RigidTy::Array(element, _)) = current_ty.kind() else {
                    return None;
                };
                current_ty = element;
            }
            _ => return None,
        }
    }

    if enum_variant.is_some() {
        return None;
    }

    Some(ResolvedDebugProjection {
        dereference_base,
        offset_bytes,
        ty: current_ty,
    })
}

fn debug_source_position(span: rustc_public::ty::Span) -> Option<DebugSourcePosition> {
    let file = span.get_filename();
    let lines = span.get_lines();
    if file.is_empty() || lines.start_line == 0 || lines.start_col == 0 {
        return None;
    }
    Some(DebugSourcePosition {
        file: std::path::PathBuf::from(file),
        line: lines.start_line as i32,
        column: lines.start_col as i32,
    })
}

/// Source-level names for MIR locals, independent of the selected debug tier.
///
/// Full variable debug information is deliberately optional, but the local
/// memory diagnostic still needs a useful source identity in optimized builds.
/// `var_debug_info` is already available in stable MIR and does not force LLVM
/// debug metadata emission, so keep this lightweight map separate from
/// [`collect_debug_locals`].
fn collect_local_source_names(body: &mir::Body) -> FxHashMap<mir::Local, String> {
    let mut names = FxHashMap::default();
    for info in &body.var_debug_info {
        let local = match &info.value {
            mir::VarDebugInfoContents::Place(place) if place.projection.is_empty() => place.local,
            mir::VarDebugInfoContents::Place(_) | mir::VarDebugInfoContents::Const(_) => continue,
        };
        let name = info.name.to_string();
        if !name.is_empty() {
            names.entry(local).or_insert(name);
        }
    }
    names
}

/// Compact source-level type spelling for local-memory diagnostics.
fn local_memory_type_name(ty: &Ty) -> String {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool".to_string(),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty).to_string(),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty).to_string(),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty).to_string(),
        TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
            raw_pointer_name(pointee, mutability)
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
            reference_name(pointee, mutability)
        }
        TyKind::RigidTy(RigidTy::Tuple(subtypes)) => format!(
            "({})",
            subtypes
                .iter()
                .map(local_memory_type_name)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        TyKind::RigidTy(RigidTy::Array(element, len)) => {
            let count = array_len_const(&len)
                .map(|count| count.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("[{}; {count}]", local_memory_type_name(&element))
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => adt_def.trimmed_name(),
        // Closure and coroutine environments reach here through
        // `var_debug_info` merged from MIR-inlined callees (iterator adapters
        // name their closure parameters, e.g. `f`). Their `{ty:?}` dump spells
        // DefIds and generic args recursively and can run to many kilobytes,
        // which would then be hex-encoded into an SSA value name; LLVM's
        // textual parser mis-lexes identifiers that long. Spell them the way
        // rustc diagnostics do instead.
        TyKind::RigidTy(RigidTy::Closure(..)) => "{closure}".to_string(),
        TyKind::RigidTy(
            RigidTy::Coroutine(..) | RigidTy::CoroutineClosure(..) | RigidTy::CoroutineWitness(..),
        ) => "{coroutine}".to_string(),
        _ => bounded_type_spelling(ty),
    }
}

/// Debug-format spelling for type kinds without a dedicated compact arm,
/// hard-capped in length.
///
/// The spelling exists to be read in a one-line warning and travels inside an
/// SSA value name, so an unbounded `{ty:?}` dump is never acceptable here even
/// for kinds this function does not anticipate.
fn bounded_type_spelling(ty: &Ty) -> String {
    const MAX_TYPE_SPELLING_BYTES: usize = 64;
    let mut spelled = format!("{ty:?}");
    if spelled.len() > MAX_TYPE_SPELLING_BYTES {
        let mut cut = MAX_TYPE_SPELLING_BYTES;
        while !spelled.is_char_boundary(cut) {
            cut -= 1;
        }
        spelled.truncate(cut);
        spelled.push_str("...");
    }
    spelled
}

/// Describe one MIR local as the provenance attribute carried by `mir.alloca`.
///
/// The attribute stays a first-class IR citizen through lowering; only the
/// textual LLVM exporter serializes it (hex-encoded into the alloca's SSA
/// name), so arbitrary Rust identifiers and type spellings cannot make
/// invalid LLVM IR.
fn local_memory_provenance(local_idx: usize, name: &str, ty: &Ty) -> LocalMemoryProvenanceAttr {
    let size_bytes = ty
        .layout()
        .ok()
        .map(|layout| layout.shape().size.bytes() as u64)
        .unwrap_or(0);
    LocalMemoryProvenanceAttr {
        local_index: local_idx as u64,
        size_bytes,
        binding_name: name.into(),
        type_name: local_memory_type_name(ty).into(),
    }
}

/// Maximum nesting depth for composite debug types. Guards against deeply
/// nested or (via generics) pathological value-type trees; beyond this we omit
/// the inner detail rather than recurse without bound.
const MAX_DEBUG_TYPE_DEPTH: usize = 8;

fn debug_type_for_ty(ty: &Ty) -> Option<DebugLocalTypeKind> {
    debug_type_for_ty_at(ty, 0)
}

fn debug_type_for_ty_at(ty: &Ty, depth: usize) -> Option<DebugLocalTypeKind> {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => Some(DebugLocalTypeKind::Basic {
            name: "bool".to_string(),
            size_bits: 8,
            encoding: "DW_ATE_boolean",
        }),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => Some(DebugLocalTypeKind::Basic {
            name: int_name(int_ty).to_string(),
            size_bits: (int_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_signed",
        }),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => Some(DebugLocalTypeKind::Basic {
            name: uint_name(uint_ty).to_string(),
            size_bits: (uint_ty.num_bytes() * 8) as u64,
            encoding: "DW_ATE_unsigned",
        }),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => Some(DebugLocalTypeKind::Basic {
            name: float_name(float_ty).to_string(),
            size_bits: float_size_bits(float_ty),
            encoding: "DW_ATE_float",
        }),
        TyKind::RigidTy(RigidTy::RawPtr(pointee, mutability)) => {
            Some(DebugLocalTypeKind::Pointer {
                name: raw_pointer_name(pointee, mutability),
                size_bits: 64,
            })
        }
        TyKind::RigidTy(RigidTy::Ref(_, pointee, mutability)) => {
            Some(DebugLocalTypeKind::Pointer {
                name: reference_name(pointee, mutability),
                size_bits: 64,
            })
        }
        TyKind::RigidTy(RigidTy::Closure(closure_def, substs)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let upvar_tys = types::closure_upvar_tys(&substs)?;
            let fields = upvar_tys
                .into_iter()
                .enumerate()
                .map(|(idx, upvar_ty)| (format!("capture_{idx}"), upvar_ty));
            debug_struct_type(ty, format!("{:?}", closure_def.def_id()), fields, depth)
        }
        TyKind::RigidTy(RigidTy::Tuple(subtypes)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let name = format!(
                "({})",
                subtypes
                    .iter()
                    .map(short_ty_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let fields = subtypes
                .iter()
                .enumerate()
                .map(|(idx, sub)| (format!("__{idx}"), *sub));
            debug_struct_type(ty, name, fields, depth)
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            match adt_def.kind() {
                rustc_public::ty::AdtKind::Struct => {
                    let variants = adt_def.variants();
                    if variants.len() != 1 {
                        return None;
                    }
                    let name = adt_def.trimmed_name();
                    let fields = variants[0]
                        .fields()
                        .into_iter()
                        .map(|field| (field.name.to_string(), field.ty_with_args(&substs)));
                    debug_struct_type(ty, name, fields, depth)
                }
                rustc_public::ty::AdtKind::Enum => debug_enum_type(ty, depth),
                rustc_public::ty::AdtKind::Union => None,
            }
        }
        TyKind::RigidTy(RigidTy::Array(elem_ty, len_const)) if depth < MAX_DEBUG_TYPE_DEPTH => {
            let count = array_len_const(&len_const)?;
            let element = debug_type_for_ty_at(&elem_ty, depth + 1)?;
            let size_bits = layout_size_bits(ty)?;
            Some(DebugLocalTypeKind::Array {
                name: format!("[{}; {count}]", short_ty_name(&elem_ty)),
                size_bits,
                element: Box::new(element),
                count,
            })
        }
        _ => None,
    }
}

/// Build a `DICompositeType`-shaped struct/tuple from rustc's real layout.
///
/// Member offsets come from `ty.layout()` (so `repr(Rust)` field reordering is
/// honored), not declaration order. Fields whose type we cannot yet describe,
/// and zero-sized fields (e.g. `PhantomData`), are omitted; the remaining
/// members keep their correct offsets.
fn debug_struct_type(
    ty: &Ty,
    name: String,
    fields: impl Iterator<Item = (String, Ty)>,
    depth: usize,
) -> Option<DebugLocalTypeKind> {
    let layout = ty.layout().ok()?;
    let shape = layout.shape();
    let offsets: Vec<u64> = match &shape.fields {
        rustc_public::abi::FieldsShape::Arbitrary { offsets } => {
            offsets.iter().map(|off| off.bytes() as u64).collect()
        }
        _ => return None,
    };
    let size_bits = shape.size.bytes() as u64 * 8;

    let mut members = Vec::new();
    for (idx, (field_name, field_ty)) in fields.enumerate() {
        let offset_bytes = *offsets.get(idx)?;
        let Some(member_ty) = debug_type_for_ty_at(&field_ty, depth + 1) else {
            continue;
        };
        if member_ty.size_bits() == 0 {
            continue;
        }
        members.push(DebugTypeMember {
            name: field_name,
            offset_bits: offset_bytes * 8,
            ty: member_ty,
        });
    }

    if members.is_empty() {
        return None;
    }

    Some(DebugLocalTypeKind::Struct {
        name,
        size_bits,
        members,
    })
}

/// Build a Rust enum debug type using rustc's physical layout.
///
/// This mirrors rustc's native DWARF representation rather than guessing from
/// source-level `Option`/`Result` shapes: a top-level structure contains a
/// variant part, whose discriminator is either the direct integer tag or the
/// integer-normalized niche carrier. Variant payload fields use rustc's exact
/// per-variant offsets. For niche layouts the untagged variant deliberately has
/// no discriminant value, so the debugger treats it as the default branch.
fn debug_enum_type(ty: &Ty, depth: usize) -> Option<DebugLocalTypeKind> {
    let TyKind::RigidTy(RigidTy::Adt(adt_def, substs)) = ty.kind() else {
        return None;
    };
    if !matches!(adt_def.kind(), rustc_public::ty::AdtKind::Enum) {
        return None;
    }

    #[derive(Clone, Copy)]
    enum DebugEnumLayout {
        Direct {
            width: u64,
        },
        Niche {
            width: u64,
            niche_variant_start: usize,
            niche_start: u128,
            untagged_variant: usize,
        },
        Single,
        Empty,
    }

    let layout_shape = ty.layout().ok()?.shape();
    let size_bits = layout_shape.size.bytes() as u64 * 8;

    let (discriminant, debug_layout) = match &layout_shape.variants {
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding: rustc_public::abi::TagEncoding::Direct,
            tag_field,
            ..
        } => {
            let primitive = match tag {
                rustc_public::abi::Scalar::Initialized { value, .. }
                | rustc_public::abi::Scalar::Union { value } => *value,
            };
            let rustc_public::abi::Primitive::Int { length, signed } = primitive else {
                return None;
            };
            let width = length.bits() as u64;
            if width == 0 || width > 64 {
                return None;
            }
            let offset_bits = crate::translator::layout::enum_tag_offset(
                &layout_shape.fields,
                *tag_field,
                pliron::location::Location::Unknown,
            )
            .ok()? as u64
                * 8;
            let tag_ty = DebugLocalTypeKind::Basic {
                name: format!("{}{}", if signed { "i" } else { "u" }, width),
                size_bits: width,
                encoding: if signed {
                    "DW_ATE_signed"
                } else {
                    "DW_ATE_unsigned"
                },
            };
            (
                Some(DebugEnumDiscriminant {
                    offset_bits,
                    ty: Box::new(tag_ty),
                }),
                DebugEnumLayout::Direct { width },
            )
        }
        rustc_public::abi::VariantsShape::Multiple {
            tag,
            tag_encoding:
                rustc_public::abi::TagEncoding::Niche {
                    untagged_variant,
                    niche_variants,
                    niche_start,
                },
            tag_field,
            ..
        } => {
            let primitive = match tag {
                rustc_public::abi::Scalar::Initialized { value, .. }
                | rustc_public::abi::Scalar::Union { value } => *value,
            };
            let width = primitive
                .size(&rustc_public::target::MachineInfo::target())
                .bits() as u64;
            if width == 0 || width > 64 {
                return None;
            }
            let offset_bits = crate::translator::layout::enum_tag_offset(
                &layout_shape.fields,
                *tag_field,
                pliron::location::Location::Unknown,
            )
            .ok()? as u64
                * 8;

            // rustc normalizes niche carriers, including pointer niches, to an
            // unsigned integer of the same physical width for DWARF.
            let tag_name = match primitive {
                rustc_public::abi::Primitive::Pointer(_) if width == 64 => "usize".to_string(),
                _ => format!("u{width}"),
            };
            let tag_ty = DebugLocalTypeKind::Basic {
                name: tag_name,
                size_bits: width,
                encoding: "DW_ATE_unsigned",
            };
            (
                Some(DebugEnumDiscriminant {
                    offset_bits,
                    ty: Box::new(tag_ty),
                }),
                DebugEnumLayout::Niche {
                    width,
                    niche_variant_start: niche_variants.start().to_index(),
                    niche_start: *niche_start,
                    untagged_variant: untagged_variant.to_index(),
                },
            )
        }
        rustc_public::abi::VariantsShape::Single { .. } => (None, DebugEnumLayout::Single),
        rustc_public::abi::VariantsShape::Empty => (None, DebugEnumLayout::Empty),
    };

    let source_variants = adt_def.variants();
    let mut variants = Vec::with_capacity(source_variants.len());

    for (variant_index, variant) in source_variants.iter().enumerate() {
        let fields = variant.fields();
        let field_offsets: Vec<u64> = match &layout_shape.variants {
            rustc_public::abi::VariantsShape::Single { index }
                if index.to_index() != variant_index =>
            {
                vec![0; fields.len()]
            }
            rustc_public::abi::VariantsShape::Empty => vec![0; fields.len()],
            _ => crate::translator::layout::enum_variant_field_offsets(
                &layout_shape,
                variant_index,
                pliron::location::Location::Unknown,
            )
            .ok()?
            .into_iter()
            .map(|offset| offset as u64)
            .collect(),
        };

        let mut members = Vec::new();
        for (field_index, field) in fields.into_iter().enumerate() {
            let field_ty = field.ty_with_args(&substs);
            let Some(member_ty) = debug_type_for_ty_at(&field_ty, depth + 1) else {
                continue;
            };
            if member_ty.size_bits() == 0 {
                continue;
            }
            let offset_bytes = *field_offsets.get(field_index)?;
            let source_name = field.name.to_string();
            let member_name = if source_name.parse::<usize>().ok() == Some(field_index) {
                format!("__{field_index}")
            } else {
                source_name
            };
            members.push(DebugTypeMember {
                name: member_name,
                offset_bits: offset_bytes * 8,
                ty: member_ty,
            });
        }

        let discriminant_value = match debug_layout {
            DebugEnumLayout::Direct { width } => {
                let variant_idx = rustc_public::ty::VariantIdx::to_val(variant_index);
                let raw = adt_def.discriminant_for_variant(variant_idx).val;
                truncate_debug_discriminant(raw, width)
            }
            DebugEnumLayout::Niche {
                width,
                niche_variant_start,
                niche_start,
                untagged_variant,
            } => {
                if variant_index == untagged_variant {
                    None
                } else {
                    let raw = (variant_index as u128)
                        .wrapping_sub(niche_variant_start as u128)
                        .wrapping_add(niche_start);
                    truncate_debug_discriminant(raw, width)
                }
            }
            DebugEnumLayout::Single | DebugEnumLayout::Empty => None,
        };

        variants.push(DebugEnumVariant {
            name: variant.name().to_string(),
            discriminant: discriminant_value,
            members,
        });
    }

    Some(DebugLocalTypeKind::Enum {
        name: adt_def.trimmed_name(),
        size_bits,
        discriminant,
        variants,
    })
}

/// Truncate a physical discriminant to the width LLVM will attach as
/// `extraData` on the corresponding variant member.
fn truncate_debug_discriminant(value: u128, width: u64) -> Option<u64> {
    if width == 0 || width > 64 {
        return None;
    }
    let mask = if width == 64 {
        u128::from(u64::MAX)
    } else {
        (1u128 << width) - 1
    };
    Some((value & mask) as u64)
}

/// Total size of `ty` in bits from its layout, or `None` if unavailable.
fn layout_size_bits(ty: &Ty) -> Option<u64> {
    Some(ty.layout().ok()?.shape().size.bytes() as u64 * 8)
}

/// Evaluate a fixed array's length constant to a `u64`.
fn array_len_const(len_const: &rustc_public::ty::TyConst) -> Option<u64> {
    match len_const.kind() {
        rustc_public::ty::TyConstKind::Value(_, alloc) => {
            let mut arr = [0u8; 8];
            for (i, byte) in alloc.bytes.iter().take(8).enumerate() {
                arr[i] = (*byte)?;
            }
            Some(u64::from_le_bytes(arr))
        }
        _ => None,
    }
}

/// A short, human-readable name for a type, used only for composite display.
fn short_ty_name(ty: &Ty) -> String {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool".to_string(),
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty).to_string(),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty).to_string(),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty).to_string(),
        TyKind::RigidTy(RigidTy::RawPtr(..)) | TyKind::RigidTy(RigidTy::Ref(..)) => {
            "ptr".to_string()
        }
        TyKind::RigidTy(RigidTy::Adt(adt_def, _)) => adt_def.trimmed_name(),
        _ => "_".to_string(),
    }
}

fn int_name(ty: IntTy) -> &'static str {
    match ty {
        IntTy::Isize => "isize",
        IntTy::I8 => "i8",
        IntTy::I16 => "i16",
        IntTy::I32 => "i32",
        IntTy::I64 => "i64",
        IntTy::I128 => "i128",
    }
}

fn uint_name(ty: UintTy) -> &'static str {
    match ty {
        UintTy::Usize => "usize",
        UintTy::U8 => "u8",
        UintTy::U16 => "u16",
        UintTy::U32 => "u32",
        UintTy::U64 => "u64",
        UintTy::U128 => "u128",
    }
}

fn float_name(ty: FloatTy) -> &'static str {
    match ty {
        FloatTy::F16 => "f16",
        FloatTy::F32 => "f32",
        FloatTy::F64 => "f64",
        FloatTy::F128 => "f128",
    }
}

fn float_size_bits(ty: FloatTy) -> u64 {
    match ty {
        FloatTy::F16 => 16,
        FloatTy::F32 => 32,
        FloatTy::F64 => 64,
        FloatTy::F128 => 128,
    }
}

fn raw_pointer_name(pointee: Ty, mutability: mir::Mutability) -> String {
    let mutability = match mutability {
        mir::Mutability::Mut => "mut ",
        mir::Mutability::Not => "const ",
    };
    format!("*{mutability}{}", simple_type_name(&pointee))
}

fn reference_name(pointee: Ty, mutability: mir::Mutability) -> String {
    let mutability = match mutability {
        mir::Mutability::Mut => "mut ",
        mir::Mutability::Not => "",
    };
    format!("&{mutability}{}", simple_type_name(&pointee))
}

fn simple_type_name(ty: &Ty) -> &'static str {
    match ty.kind() {
        TyKind::RigidTy(RigidTy::Bool) => "bool",
        TyKind::RigidTy(RigidTy::Int(int_ty)) => int_name(int_ty),
        TyKind::RigidTy(RigidTy::Uint(uint_ty)) => uint_name(uint_ty),
        TyKind::RigidTy(RigidTy::Float(float_ty)) => float_name(float_ty),
        _ => "_",
    }
}

/// Emit one `mir.alloca` per non-ZST MIR local at the top of the entry block,
/// then store each function argument into its backing slot.
///
/// This is the foundation of the alloca + load/store translator model: every
/// non-ZST MIR local is backed by a single stack slot recorded in `value_map`
/// via [`ValueMap::set_slot`]. Function arguments (which arrive as entry-block
/// arguments) are immediately stored into their slots so subsequent blocks can
/// load them without needing SSA block arguments.
///
/// `num_args` is the number of function arguments (MIR locals `1..=num_args`).
///
/// Returns the last operation emitted, so the caller can pass it to
/// [`block::translate_block`] as `entry_prev_op` to append block contents
/// **after** this setup (otherwise `insert_at_front` would push the alloca
/// chain past the block terminator).
///
/// # ZST locals
///
/// Locals whose Rust type is zero-sized (unit tuple, empty structs, `!`, …)
/// are skipped entirely: they get no slot in [`ValueMap`] and any attempted
/// load/store short-circuits.
///
/// # Unsupported types
///
/// [`types::translate_type`] can fail for locals whose types aren't supported
/// yet (e.g. ghost locals in kernels targeting unsupported surfaces). Those
/// locals simply get no slot; any later attempt to use them still errors out
/// through the existing unsupported-type code paths.
fn emit_entry_allocas(
    ctx: &mut Context,
    body: &mir::Body,
    entry_block: Ptr<BasicBlock>,
    num_args: usize,
    value_map: &mut ValueMap,
    debug_kind: DebugKind,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
    reachable: &std::collections::BTreeSet<usize>,
) -> Option<Ptr<Operation>> {
    let mut prev_op: Option<Ptr<Operation>> = None;
    let debug_locals = if debug_kind.variables_enabled() {
        collect_debug_locals(ctx, body)
    } else {
        CollectedDebugLocals::default()
    };
    let local_source_names = collect_local_source_names(body);

    // Translate local types once up front. The address-space analyzer uses
    // each pointer local's declared lowering as the conservative fallback for
    // writes it cannot classify, and the allocation loop reuses the same
    // handles below.
    let mut mir_types = Vec::with_capacity(body.locals().len());
    for local_decl in body.locals() {
        let mir_ty = if types::is_rust_type_zst(&local_decl.ty) {
            None
        } else {
            types::translate_type(ctx, &local_decl.ty).ok()
        };
        mir_types.push(mir_ty);
    }
    let declared_addr_spaces: Vec<Option<u32>> = mir_types
        .iter()
        .map(|mir_ty| {
            mir_ty
                .as_ref()
                .and_then(|mir_ty| values::pointer_addr_space(ctx, *mir_ty))
        })
        .collect();

    // Pre-scan only rustc-reachable writes. A slot is narrowed to a concrete
    // address space only when every reachable write agrees; unknown writes
    // retain their declared lowering (normally generic address space zero).
    let slot_addr_spaces =
        SlotAddrSpaceMap::analyze(body, reachable, num_args, &declared_addr_spaces);

    for local_idx in 0..body.locals().len() {
        let local = mir::Local::from(local_idx);
        let Some(mir_ty) = mir_types[local_idx] else {
            continue;
        };

        // Override the Rust-declared addrspace with the inferred one for
        // pointer slots. Non-pointer slots are untouched by
        // `align_pointer_addr_space`.
        let rust_declared = declared_addr_spaces[local_idx].unwrap_or(address_space::GENERIC);
        let target = slot_addr_spaces.effective(local, rust_declared);
        let mir_ty = values::align_pointer_addr_space(ctx, mir_ty, target);

        let (op, slot) = ValueMap::emit_alloca(ctx, mir_ty, entry_block, prev_op);

        // Tag only named Rust source locals. Compiler temporaries and lowering-
        // synthesized LLVM allocas must not turn verbose builds into a stream of
        // warnings that cannot be attributed back to user code.
        if let Some(source_name) = local_source_names.get(&local)
            && let Some(decl) = body.local_decl(local)
        {
            llvm_export::ops::set_local_memory_provenance(
                ctx,
                op,
                local_memory_provenance(local_idx, source_name, &decl.ty),
            );
        }

        if let Some(info) = debug_locals.whole.get(&local) {
            llvm_export::ops::set_debug_local_variable(ctx, op, info.variable.clone());
            if debug_source_scopes
                .is_some_and(|map| map.scopes.iter().any(|scope| scope.id == info.source_scope))
            {
                llvm_export::ops::set_debug_local_source_scope(ctx, op, info.source_scope);
            }
            op.deref_mut(ctx).set_loc(info.loc.clone());
        }
        if let Some(projected) = debug_locals.projected.get(&local) {
            llvm_export::ops::set_debug_projected_variables(ctx, op, projected);
        }
        if let Some(fragments) = debug_locals.fragments.get(&local) {
            llvm_export::ops::set_debug_fragment_variables(ctx, op, fragments);
        }
        prev_op = Some(op);
        value_map.set_slot(local, slot);
    }

    for arg_idx in 0..num_args {
        let local = mir::Local::from(arg_idx + 1);
        let block_arg = entry_block.deref(ctx).get_argument(arg_idx);
        if let Some(op) = value_map.store_local(ctx, local, block_arg, entry_block, prev_op) {
            prev_op = Some(op);
        }
    }

    prev_op
}

/// Translates a MIR function body to a pliron IR `mir.func` operation.
///
/// # Process
///
/// 1. Extract signature (arg types from MIR locals 1..N, return from local 0)
/// 2. Create `mir.func` with signature and optional `gpu_kernel` attribute
/// 3. Create one pliron block per MIR block. The entry block carries the
///    function parameters; every other block is argument-less (cross-block
///    data flow travels through per-local alloca slots)
/// 4. Emit one `mir.alloca` per non-ZST local at the top of the entry block
///    and seed the argument slots from the block's parameters
/// 5. Translate every reachable block in index order
///
/// # Arguments
///
/// * `ctx` - Pliron IR context
/// * `body` - MIR function body
/// * `instance` - Monomorphized instance (with concrete generic args)
/// * `rustc_mir_block_count` - Block count recorded from the rustc MIR body
///   before conversion to public MIR
/// * `rustc_mono_successors` - Exact per-block successor edges computed by
///   rustc's monomorphization rules under the device runtime-check policy
/// * `is_kernel` - Add `gpu_kernel` attribute for kernel entry points
/// * `is_inline_always` - Add `alwaysinline` attribute (non-kernel functions
///   marked `#[inline(always)]` in rustc)
/// * `override_name` - Custom export name (defaults to instance name)
pub fn translate_body(
    ctx: &mut Context,
    body: &mir::Body,
    instance: &mono::Instance,
    rustc_mir_block_count: usize,
    rustc_mono_successors: &[Vec<usize>],
    is_kernel: bool,
    is_inline_always: bool,
    override_name: Option<&str>,
    legaliser: &mut Legaliser,
    debug_kind: DebugKind,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
) -> TranslationResult<Ptr<Operation>> {
    // Establish and validate rustc's exact per-instance reachability before
    // any whole-body semantic scan. Dead blocks must not influence function
    // attributes, pointer-slot address spaces, or later code emission.
    if let Err(error) =
        validate_monomorphized_successors(body, rustc_mir_block_count, rustc_mono_successors)
    {
        return input_err_noloc!(TranslationErr::invalid_op(error));
    }
    let reachable = compute_reachable_blocks(body, rustc_mono_successors);

    // Create a value map to track MIR locals -> pliron IR values
    let num_locals = body.locals().len();
    let mut value_map = ValueMap::new(num_locals);

    // Resolve the per-body unchecked-indexing policy. Like the dynamic-shared
    // marker, the `#[kernel(unchecked_indexing)]` marker is scanned on any
    // function: generic kernel expansion forwards it to the generated entry
    // but also keeps the original in the `#[inline(always)]` implementation
    // helper, and either body may be the one translated here. The whole-build
    // environment switch additionally covers separately translated
    // `#[device]` functions that carry no marker.
    let unchecked_indexing = match detect_unchecked_indexing_config(body, &reachable) {
        Ok(marker_enabled) => marker_enabled || unchecked_indexing_env_enabled(),
        Err(error) => {
            return input_err_noloc!(TranslationErr::invalid_op(error));
        }
    };
    value_map.set_unchecked_indexing(unchecked_indexing);
    if unchecked_indexing && std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
        eprintln!("  Unchecked indexing enabled: bounds-check asserts elided");
    }

    // Get function argument types for the first block
    // In MIR, locals[0] is the return value, locals[1..arg_count+1] are function arguments
    let mut arg_types = Vec::new();

    // Determine argument count from the function type in the instance
    // Get the function signature to determine the number of arguments
    let fn_ty = instance.ty();
    let num_args = match fn_ty.kind() {
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::FnDef(_, _)) => {
            // Get the function signature from fn_sig()
            let sig_binder = fn_ty.kind().fn_sig().unwrap();
            // Skip the binder to get the actual signature
            let sig = sig_binder.skip_binder();
            let inputs = sig.inputs();
            inputs.len()
        }
        rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Closure(_, _)) => {
            // Closures use RustCall ABI where:
            // - MIR local 1 = self (closure environment, even if ZST)
            // - MIR locals 2..N = unpacked arguments from the fn_sig's tuple input
            //
            // fn_sig().inputs() returns just the tuple, NOT including self.
            // We need to count: 1 (self) + unpacked tuple elements
            let sig_binder = fn_ty.kind().fn_sig().unwrap();
            let sig = sig_binder.skip_binder();
            let inputs = sig.inputs();

            // The input should be a single tuple (RustCall convention)
            let tuple_arg_count = if inputs.len() == 1 {
                // Get the tuple type and count its elements
                if let rustc_public::ty::TyKind::RigidTy(rustc_public::ty::RigidTy::Tuple(
                    tuple_tys,
                )) = inputs[0].kind()
                {
                    tuple_tys.len()
                } else {
                    // Not a tuple - use 1 (single arg)
                    1
                }
            } else {
                // Multiple inputs (shouldn't happen with RustCall, but handle it)
                inputs.len()
            };

            // Total args = 1 (self) + unpacked tuple elements
            1 + tuple_arg_count
        }
        _ => {
            return input_err_noloc!(TranslationErr::unsupported(format!(
                "Expected FnDef or Closure type for function, got {:?}",
                fn_ty.kind()
            )));
        }
    };

    for arg_idx in 0..num_args {
        // MIR local index for arguments: local 1, 2, 3, ... (0 is return value)
        let local = mir::Local::from(arg_idx + 1);
        let local_decl = &body.locals()[local];
        let ty = &local_decl.ty;
        let arg_type = types::translate_type(ctx, ty)?;
        arg_types.push(arg_type);
    }

    // Get return type (local 0)
    let return_local = mir::Local::from(0usize);
    let return_decl = &body.locals()[return_local];
    let return_type_ptr = types::translate_type(ctx, &return_decl.ty)?;

    // Unit-tuple returns become a void `mir.func` signature. We skip the
    // result so `MirReturnOp` isn't expected to carry an unused `()` operand
    // (`mir-lower` reconstructs the unit value at LLVM lowering time).
    let is_unit_return = {
        let return_type_obj = return_type_ptr.deref(ctx);
        if let Some(tuple_ty) = return_type_obj.downcast_ref::<dialect_mir::types::MirTupleType>() {
            tuple_ty.get_types().is_empty()
        } else {
            false
        }
    };

    let return_types = if is_unit_return {
        vec![]
    } else {
        vec![return_type_ptr]
    };

    // Create function type for signature
    use pliron::builtin::attributes::TypeAttr;
    use pliron::builtin::types::FunctionType;
    let func_type = FunctionType::get(
        ctx,
        arg_types.clone(), // inputs
        return_types,      // results
    );
    let func_type_attr = TypeAttr::new(func_type.into());

    // Create a mir.func operation with a region for the function body
    let op_ptr = Operation::new(
        ctx,
        MirFuncOp::get_concrete_op_info(),
        vec![], // No result types
        vec![], // No operands
        vec![], // No successors
        1,      // 1 region for function body
    );

    // Set the function location from rustc's body span. This becomes the
    // default scope line for line-table debug info once LLVM export is enabled.
    let loc = span_to_location(ctx, body.span);
    op_ptr.deref_mut(ctx).set_loc(loc);

    // Create MirFuncOp and set the function type attribute and symbol name
    let mir_func_op = MirFuncOp::new(ctx, op_ptr, func_type_attr);

    let name_str = if let Some(name) = override_name {
        name.to_string()
    } else {
        instance.name().to_string()
    };
    mir_func_op.set_symbol_name(ctx, legaliser.legalise(&name_str));

    // Check if the function has the #[cuda_oxide::kernel] attribute (passed via is_kernel flag)
    if is_kernel {
        // Add "gpu_kernel" attribute to the mir.func operation.
        // This will be used by the lowering pass to set the "gpu_kernel" attribute on the llvm.func.
        use pliron::builtin::attributes::StringAttr;
        let kernel_attr = StringAttr::new("true".to_string());
        let key: Identifier = "gpu_kernel".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, kernel_attr);

        // Detect compile-time cluster configuration from #[cluster(x,y,z)] attribute
        if let Some(cluster_dims) = detect_cluster_config(body, &reachable) {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            // Add cluster_dim_x/y/z attributes
            // These will be used by the LLVM export to emit nvvm.annotations metadata
            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            // Create APInt values for each dimension
            let apint_x = APInt::from_u32(cluster_dims.x, width);
            let apint_y = APInt::from_u32(cluster_dims.y, width);
            let apint_z = APInt::from_u32(cluster_dims.z, width);

            let x_attr = IntegerAttr::new(u32_ty, apint_x);
            let y_attr = IntegerAttr::new(u32_ty, apint_y);
            let z_attr = IntegerAttr::new(u32_ty, apint_z);

            let x_key: Identifier = "cluster_dim_x".try_into().unwrap();
            let y_key: Identifier = "cluster_dim_y".try_into().unwrap();
            let z_key: Identifier = "cluster_dim_z".try_into().unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            op_mut.attributes.set(x_key, x_attr);
            op_mut.attributes.set(y_key, y_attr);
            op_mut.attributes.set(z_key, z_attr);

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                eprintln!(
                    "  Cluster config detected: {}x{}x{}",
                    cluster_dims.x, cluster_dims.y, cluster_dims.z
                );
            }
        }

        // Detect compile-time launch bounds from #[launch_bounds(max, min)] attribute
        let launch_bounds = match detect_launch_bounds_config(body, &reachable) {
            Ok(bounds) => bounds,
            Err(error) => {
                return input_err_noloc!(TranslationErr::invalid_op(error));
            }
        };

        // Detect the exact block shape from #[launch_contract(block = (x,y,z))].
        // The exporter emits this as reqntid and suppresses maxntid, which ptxas
        // rejects alongside it.
        let contract_block = match detect_contract_block_config(body, &reachable) {
            Ok(block) => block,
            Err(error) => {
                return input_err_noloc!(TranslationErr::invalid_op(error));
            }
        };

        if let (Some(bounds), Some(block)) = (launch_bounds, contract_block)
            && let Err(error) = validate_block_against_bounds(bounds, block)
        {
            return input_err_noloc!(TranslationErr::invalid_op(error));
        }

        if let Some(launch_bounds) = launch_bounds {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            // Add maxntid and minctasm attributes
            // These will be used by the LLVM export to emit nvvm.annotations metadata
            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            // Create APInt values
            let apint_max = APInt::from_u32(launch_bounds.max_threads, width);
            let max_attr = IntegerAttr::new(u32_ty, apint_max);
            let max_key: Identifier = "maxntid".try_into().unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            op_mut.attributes.set(max_key, max_attr);

            // Only add minctasm if it's non-zero (specified)
            if launch_bounds.min_blocks > 0 {
                let apint_min = APInt::from_u32(launch_bounds.min_blocks, width);
                let min_attr = IntegerAttr::new(u32_ty, apint_min);
                let min_key: Identifier = "minctasm".try_into().unwrap();
                op_mut.attributes.set(min_key, min_attr);
            }

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                if launch_bounds.min_blocks > 0 {
                    eprintln!(
                        "  Launch bounds detected: maxntid={}, minctasm={}",
                        launch_bounds.max_threads, launch_bounds.min_blocks
                    );
                } else {
                    eprintln!(
                        "  Launch bounds detected: maxntid={}",
                        launch_bounds.max_threads
                    );
                }
            }
        }

        if let Some(contract_block) = contract_block {
            use pliron::builtin::attributes::IntegerAttr;
            use pliron::builtin::types::Signedness;
            use pliron::utils::apint::APInt;
            use std::num::NonZero;

            let u32_ty = pliron::builtin::types::IntegerType::get(ctx, 32, Signedness::Unsigned);
            let width = NonZero::new(32).unwrap();

            let mut op_mut = mir_func_op.get_operation().deref_mut(ctx);
            for (key, value) in [
                ("reqntid_x", contract_block.x),
                ("reqntid_y", contract_block.y),
                ("reqntid_z", contract_block.z),
            ] {
                let attr = IntegerAttr::new(u32_ty, APInt::from_u32(value, width));
                let key: Identifier = key.try_into().unwrap();
                op_mut.attributes.set(key, attr);
            }

            if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
                eprintln!(
                    "  Launch contract block detected: reqntid={}x{}x{}",
                    contract_block.x, contract_block.y, contract_block.z
                );
            }
        }
    }

    // Attribute macros may run before `#[kernel]`. Generic expansion forwards
    // that marker to the entry but also keeps the original in its helper, so
    // record markers on any function. mir-lower treats every marked local
    // function as a propagation root and carries the minimum to its callees.
    if let Some(alignment) = detect_dynamic_shared_alignment(body, &reachable) {
        use pliron::builtin::attributes::IntegerAttr;
        use pliron::builtin::types::Signedness;
        use pliron::utils::apint::APInt;
        use std::num::NonZero;

        let u64_ty = pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Unsigned);
        let value = APInt::from_u64(alignment.bytes, NonZero::new(64).unwrap());
        let key: Identifier = "dynamic_shared_alignment".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, IntegerAttr::new(u64_ty, value));

        if std::env::var("CUDA_OXIDE_VERBOSE").is_ok() {
            eprintln!(
                "  Dynamic shared-memory contract alignment detected: {}",
                alignment.bytes
            );
        }
    }

    if let Some(scope_map) = debug_source_scopes
        && debug_kind.variables_enabled()
    {
        llvm_export::ops::set_debug_source_scope_map(ctx, op_ptr, scope_map);
    }

    set_alwaysinline_attr_from_flag(ctx, &mir_func_op, is_kernel, is_inline_always);

    // Get the function body region (region 0)
    let region_ptr = op_ptr.deref(ctx).get_region(0);

    // -------------------------------------------------------------------------
    // PHASE 1: Create all pliron IR blocks
    // -------------------------------------------------------------------------
    //
    // Only the entry block receives block arguments (the function's formal
    // parameters). Every other block is argument-less: cross-block data flow
    // travels through the per-local alloca slots, not block arguments.
    let mut block_map: Vec<Ptr<BasicBlock>> = Vec::new();

    for (idx, _mir_block) in body.blocks.iter().enumerate() {
        let arg_types_for_block = if idx == 0 { arg_types.clone() } else { vec![] };

        let block_ptr = BasicBlock::new(ctx, None, arg_types_for_block);
        block_map.push(block_ptr);
    }

    // Link all blocks into the function's region.
    for (idx, block_ptr) in block_map.iter().enumerate() {
        if idx == 0 {
            block_ptr.insert_at_front(region_ptr, ctx);
        } else {
            block_ptr.insert_after(ctx, block_map[idx - 1]);
        }
    }

    // -------------------------------------------------------------------------
    // PHASE 1.5: Entry-block allocas + argument stores
    // -------------------------------------------------------------------------
    //
    // Every non-ZST MIR local is backed by a single stack slot emitted at the
    // top of the entry block; its pointer is recorded in `value_map` via
    // `set_slot`. Function arguments are eagerly stored into their slots so
    // later blocks can `load_local` them without needing block arguments.
    //
    // The `mem2reg` pass in `pipeline.rs` promotes the scalar slots back into
    // SSA before LLVM lowering.
    let mut entry_last_op = emit_entry_allocas(
        ctx,
        body,
        block_map[0],
        num_args,
        &mut value_map,
        debug_kind,
        debug_source_scopes,
        &reachable,
    );

    let mut composite_mirror_bindings = Vec::new();
    if debug_kind == DebugKind::Full {
        let debug_locals = collect_debug_locals(ctx, body);
        entry_last_op = materialize_full_debug_constant_fragments(
            ctx,
            body,
            block_map[0],
            &mut value_map,
            entry_last_op,
            debug_locals.constant_fragments,
        )?;
        let (last_op, mirror_bindings) = materialize_full_debug_composite_mirrors(
            ctx,
            body,
            block_map[0],
            &mut value_map,
            entry_last_op,
            debug_source_scopes,
            debug_locals.composite_mirrors,
        )?;
        entry_last_op = last_op;
        composite_mirror_bindings = mirror_bindings;
    }

    // -------------------------------------------------------------------------
    // PHASE 2: Translate reachable blocks
    // -------------------------------------------------------------------------
    //
    // Every local flows through its stack slot, so blocks have no inter-block
    // ordering dependency and can be translated in a single index-order pass.
    // Unwind-only cleanup blocks are skipped here (see
    // [`non_unwind_successors`]) and patched with `mir.unreachable` below.
    let mut blocks_processed: std::collections::HashSet<usize> = std::collections::HashSet::new();

    for idx in reachable.iter().copied() {
        let mir_block = &body.blocks[idx];
        let block_ptr = block_map[idx];
        let entry_prev_op = if idx == 0 { entry_last_op } else { None };
        block::translate_block(
            ctx,
            body,
            mir_block,
            idx,
            block_ptr,
            &mut value_map,
            &block_map,
            &rustc_mono_successors[idx],
            legaliser,
            entry_prev_op,
        )?;
        blocks_processed.insert(idx);
    }

    // Unwind cleanup blocks are unreachable on GPU but pliron still requires
    // every block to have a terminator, so we stitch `mir.unreachable` onto
    // the ones we skipped above. Later passes are free to drop them as dead
    // code.
    for (idx, &block_ptr) in block_map.iter().enumerate().take(body.blocks.len()) {
        if !blocks_processed.contains(&idx) {
            let unreachable_op = Operation::new(
                ctx,
                dialect_mir::ops::MirUnreachableOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                0,
            );
            unreachable_op.insert_at_front(block_ptr, ctx);
        }
    }

    if debug_kind == DebugKind::Full {
        materialize_full_debug_composite_mirror_updates(
            ctx,
            &block_map,
            &composite_mirror_bindings,
        );
        materialize_full_debug_fragment_values(ctx, &block_map);
    }

    Ok(op_ptr)
}

/// Materialize mixed constant/local scalarized composites into one debug mirror.
///
/// NVPTX can drop one fragment of a source aggregate even when that fragment
/// is backed by a real SSA value. When MIR optimization turns one direct field
/// into a constant and another into a local, full debug therefore reconstructs
/// a single contiguous source object in debug-only memory. The mirror carries
/// one ordinary whole-variable `dbg.declare`; field writes keep it synchronized
/// with the optimized MIR locals without changing their program semantics.
fn materialize_full_debug_composite_mirrors(
    ctx: &mut Context,
    body: &mir::Body,
    entry_block: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    mut prev_op: Option<Ptr<Operation>>,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
    mirrors: Vec<CompositeDebugMirror>,
) -> TranslationResult<(Option<Ptr<Operation>>, Vec<CompositeMirrorPlaceBinding>)> {
    let mut place_bindings = Vec::new();

    for mirror in mirrors {
        let mirror_ty = types::translate_type(ctx, &mirror.source_ty)?;
        let (alloca_op, mirror_slot) = ValueMap::emit_alloca(ctx, mirror_ty, entry_block, prev_op);
        llvm_export::ops::set_debug_local_variable(ctx, alloca_op, mirror.variable.clone());
        if debug_source_scopes.is_some_and(|map| {
            map.scopes
                .iter()
                .any(|scope| scope.id == mirror.source_scope)
        }) {
            llvm_export::ops::set_debug_local_source_scope(ctx, alloca_op, mirror.source_scope);
        }
        if let Some(declaration) = &mirror.declaration {
            llvm_export::ops::set_debug_local_declaration_location(
                ctx,
                alloca_op,
                declaration.file.clone(),
                declaration.line,
                declaration.column,
            );
        }
        alloca_op.deref_mut(ctx).set_loc(mirror.loc.clone());
        prev_op = Some(alloca_op);

        for fragment in mirror.fragments {
            let field_ty = types::translate_type(ctx, &fragment.field_ty)?;
            match fragment.value {
                CompositeMirrorFragmentValue::Constant { constant, loc } => {
                    let operand = mir::Operand::Constant(constant);
                    let (value, last_inserted) = rvalue::translate_operand(
                        ctx,
                        body,
                        &operand,
                        value_map,
                        entry_block,
                        prev_op,
                        loc.clone(),
                    )?;
                    prev_op = last_inserted.or(prev_op);
                    prev_op = Some(emit_full_debug_mirror_field_store(
                        ctx,
                        entry_block,
                        prev_op,
                        mirror_slot,
                        fragment.field_index,
                        field_ty,
                        value,
                        loc,
                    ));
                }
                CompositeMirrorFragmentValue::Place { local } => {
                    if let Some(backing_slot) = value_map.get_slot(local) {
                        place_bindings.push(CompositeMirrorPlaceBinding {
                            backing_slot,
                            mirror_slot,
                            field_index: fragment.field_index,
                            field_ty,
                        });
                    }
                }
            }
        }
    }

    Ok((prev_op, place_bindings))
}

/// Store one reconstructed source field into a whole-variable debug mirror.
fn emit_full_debug_mirror_field_store(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    prev_op: Option<Ptr<Operation>>,
    mirror_slot: Value,
    field_index: usize,
    field_ty: TypeHandle,
    value: Value,
    loc: pliron::location::Location,
) -> Ptr<Operation> {
    let field_ptr_ty = MirPtrType::get_generic(ctx, field_ty, true).into();
    let field_addr_op = Operation::new(
        ctx,
        MirFieldAddrOp::get_concrete_op_info(),
        vec![field_ptr_ty],
        vec![mirror_slot],
        vec![],
        0,
    );
    field_addr_op.deref_mut(ctx).set_loc(loc.clone());
    MirFieldAddrOp::new(field_addr_op)
        .set_attr_field_index(ctx, FieldIndexAttr(field_index as u32));
    match prev_op {
        Some(prev) => field_addr_op.insert_after(ctx, prev),
        None => field_addr_op.insert_at_front(block, ctx),
    }

    let field_ptr = field_addr_op.deref(ctx).get_result(0);
    let (value, anchor) =
        values::maybe_ptr_coerce(ctx, value, field_ty, block, Some(field_addr_op));
    let store_op = Operation::new(
        ctx,
        MirStoreOp::get_concrete_op_info(),
        vec![],
        vec![field_ptr, value],
        vec![],
        0,
    );
    MirStoreOp::new(store_op).set_volatile(ctx, true);
    store_op.deref_mut(ctx).set_loc(loc);
    store_op.insert_after(ctx, anchor.unwrap_or(field_addr_op));
    store_op
}

/// Mirror every write to a scalarized backing local into its source aggregate.
fn materialize_full_debug_composite_mirror_updates(
    ctx: &mut Context,
    blocks: &[Ptr<BasicBlock>],
    bindings: &[CompositeMirrorPlaceBinding],
) {
    if bindings.is_empty() {
        return;
    }

    for &block in blocks {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some(store) = Operation::get_op::<MirStoreOp>(op, ctx) else {
                continue;
            };
            let store_address = store.address_opd(ctx);
            let store_value = store.value_opd(ctx);
            let store_loc = op.deref(ctx).loc().clone();
            let mut anchor = op;
            for &binding in bindings {
                if binding.backing_slot != store_address {
                    continue;
                }
                anchor = emit_full_debug_mirror_field_store(
                    ctx,
                    block,
                    Some(anchor),
                    binding.mirror_slot,
                    binding.field_index,
                    binding.field_ty,
                    store_value,
                    store_loc.clone(),
                );
            }
        }
    }
}

/// Materialize constant composite fragments as value-based debug locations.
///
/// MIR optimization may scalar-replace an aggregate and then constant-propagate
/// one of its pieces. rustc records that source fragment as
/// `VarDebugInfoContents::Const`; there is no backing alloca/store for the
/// later stack-slot salvage pass to observe. Emit the constant through the
/// normal operand translator and attach the fragment metadata directly to a
/// `mir.dbg_value` in the entry block.
fn materialize_full_debug_constant_fragments(
    ctx: &mut Context,
    body: &mir::Body,
    entry_block: Ptr<BasicBlock>,
    value_map: &mut ValueMap,
    mut prev_op: Option<Ptr<Operation>>,
    fragments: Vec<ConstantDebugFragment>,
) -> TranslationResult<Option<Ptr<Operation>>> {
    for fragment in fragments {
        let operand = mir::Operand::Constant(fragment.constant);
        let (value, last_inserted) = rvalue::translate_operand(
            ctx,
            body,
            &operand,
            value_map,
            entry_block,
            prev_op,
            fragment.loc.clone(),
        )?;
        prev_op = last_inserted.or(prev_op);

        let dbg_value = MirDbgValueOp::new(ctx, value);
        llvm_export::ops::set_debug_fragment_variables(
            ctx,
            dbg_value.get_operation(),
            std::slice::from_ref(&fragment.fragment),
        );
        dbg_value
            .get_operation()
            .deref_mut(ctx)
            .set_loc(fragment.loc);
        match prev_op {
            Some(prev) => dbg_value.get_operation().insert_after(ctx, prev),
            None => dbg_value.get_operation().insert_at_front(entry_block, ctx),
        }
        prev_op = Some(dbg_value.get_operation());
    }

    Ok(prev_op)
}

/// Convert fragment-backed full-debug stack slots to value locations.
///
/// rustc MIR optimization can scalar-replace one source aggregate into several
/// independent MIR locals. Keeping those pieces as separate `dbg.declare`
/// locations makes cuda-gdb treat each fragment value as an address and can
/// produce an invalid composite stack location. Full debug still keeps the
/// actual allocas in memory, but each write to a fragment-backed slot also gets
/// a `mir.dbg_value` that names the stored SSA value. The alloca fragment attrs
/// are cleared so LLVM export emits only the value-based fragment locations.
fn materialize_full_debug_fragment_values(ctx: &mut Context, blocks: &[Ptr<BasicBlock>]) {
    let mut fragment_slots = FxHashMap::default();

    for &block in blocks {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some(alloca) = Operation::get_op::<MirAllocaOp>(op, ctx) else {
                continue;
            };
            let fragments = llvm_export::ops::debug_fragment_variables(ctx, op);
            if fragments.is_empty() {
                continue;
            }

            let slot = alloca.get_operation().deref(ctx).get_result(0);
            fragment_slots.insert(slot, fragments);
            llvm_export::ops::set_debug_fragment_variables(ctx, op, &[]);
        }
    }

    if fragment_slots.is_empty() {
        return;
    }

    for &block in blocks {
        let ops: Vec<_> = block.deref(ctx).iter(ctx).collect();
        for op in ops {
            let Some(store) = Operation::get_op::<MirStoreOp>(op, ctx) else {
                continue;
            };
            let Some(fragments) = fragment_slots.get(&store.address_opd(ctx)) else {
                continue;
            };

            let dbg_value = MirDbgValueOp::new(ctx, store.value_opd(ctx));
            llvm_export::ops::set_debug_fragment_variables(
                ctx,
                dbg_value.get_operation(),
                fragments,
            );
            let loc = op.deref(ctx).loc().clone();
            dbg_value.get_operation().deref_mut(ctx).set_loc(loc);
            dbg_value.get_operation().insert_after(ctx, op);
        }
    }
}

/// Propagate `#[inline(always)]` as an LLVM `alwaysinline` function
/// attribute. Kernel entry points are excluded because they're `.entry` in PTX
/// and never callees, so marking them `alwaysinline` would be a no-op at best
/// and rejected by LLVM at worst.
fn set_alwaysinline_attr_from_flag(
    ctx: &mut Context,
    mir_func_op: &MirFuncOp,
    is_kernel: bool,
    is_inline_always: bool,
) {
    if is_inline_always && !is_kernel {
        let attr = pliron::builtin::attributes::StringAttr::new("true".to_string());
        let key: Identifier = "alwaysinline".try_into().unwrap();
        mir_func_op
            .get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, attr);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pliron::{
        basic_block::BasicBlock,
        builtin::{
            attributes::TypeAttr, op_interfaces::SymbolOpInterface, ops::ModuleOp,
            types::FunctionType,
        },
        linked_list::ContainsLinkedList,
        op::Op,
        operation::Operation,
    };

    #[test]
    fn collector_reachability_requires_the_same_public_mir_cfg() {
        let valid = [vec![2], vec![], vec![3], vec![]];
        assert!(validate_monomorphized_successor_shape(4, 4, &valid).is_ok());
        assert!(validate_monomorphized_successor_shape(4, 5, &valid).is_err());
        assert!(validate_monomorphized_successor_shape(4, 4, &valid[..3]).is_err());
        assert!(
            validate_monomorphized_successor_shape(4, 4, &[vec![4], vec![], vec![], vec![]])
                .is_err()
        );
    }

    #[test]
    fn an_exact_block_may_not_need_more_threads_than_launch_bounds_allows() {
        let block = |x, y, z| ContractBlock { x, y, z };
        let bounds = |max_threads| LaunchBounds {
            max_threads,
            min_blocks: 0,
        };

        // The shapes the `cuda_module_contract` example declares: the maximum
        // equals the required thread count on every axis product.
        assert!(validate_block_against_bounds(bounds(256), block(256, 1, 1)).is_ok());
        assert!(validate_block_against_bounds(bounds(64), block(8, 8, 1)).is_ok());
        // A maximum above the requirement is redundant, and allowed.
        assert!(validate_block_against_bounds(bounds(1024), block(16, 16, 1)).is_ok());

        // A maximum below the requirement contradicts it. Without this the
        // exporter would drop the maximum and emit the larger shape.
        let error = validate_block_against_bounds(bounds(128), block(16, 16, 1))
            .expect_err("256 threads exceed a 128-thread maximum");
        assert!(error.contains("256 threads per block"), "{error}");
        assert!(error.contains("at most 128"), "{error}");
        assert!(validate_block_against_bounds(bounds(255), block(256, 1, 1)).is_err());

        // The product is computed in u64, so a shape whose axes multiply past
        // u32 is rejected rather than wrapping to a value under the maximum.
        assert!(validate_block_against_bounds(bounds(1024), block(65_536, 65_536, 1)).is_err());
    }

    #[test]
    fn inline_always_flag_reaches_llvm_func_attr_before_export() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let module = ModuleOp::new(&mut ctx, "test_module".try_into().unwrap());
        let module_op = module.get_operation();
        let module_region = module_op.deref(&ctx).get_region(0);
        let module_block = {
            let existing = {
                let region = module_region.deref(&ctx);
                region.iter(&ctx).next()
            };
            if let Some(block) = existing {
                block
            } else {
                let block = BasicBlock::new(&mut ctx, None, vec![]);
                block.insert_at_back(module_region, &ctx);
                block
            }
        };

        let func_type = FunctionType::get(&ctx, vec![], vec![]);
        let func_type_attr = TypeAttr::new(func_type.into());
        let mir_func = {
            let op = Operation::new(
                &mut ctx,
                MirFuncOp::get_concrete_op_info(),
                vec![],
                vec![],
                vec![],
                1,
            );
            let func = MirFuncOp::new(&mut ctx, op, func_type_attr);
            func.set_symbol_name(&mut ctx, "inline_helper".try_into().unwrap());
            func
        };

        set_alwaysinline_attr_from_flag(&mut ctx, &mir_func, false, true);
        mir_func.get_operation().insert_at_back(module_block, &ctx);

        mir_lower::register(&mut ctx);
        mir_lower::lower_mir_to_llvm(&mut ctx, module_op).expect("lowering succeeds");

        let llvm_func = {
            let block = module_region.deref(&ctx).iter(&ctx).next().unwrap();
            block
                .deref(&ctx)
                .iter(&ctx)
                .find_map(|op| Operation::get_op::<llvm_export::ops::FuncOp>(op, &ctx))
                .expect("lowered LLVM function")
        };

        let key: Identifier = "alwaysinline".try_into().unwrap();
        assert!(
            llvm_func
                .get_operation()
                .deref(&ctx)
                .attributes
                .0
                .contains_key(&key),
            "`is_inline_always` must become an LLVM dialect alwaysinline attribute before export",
        );
    }

    #[test]
    fn full_debug_fragment_stores_materialize_value_locations() {
        let mut ctx = Context::new();
        crate::translator::register_dialects(&mut ctx);

        let i32_ty: pliron::r#type::TypeHandle = pliron::builtin::types::IntegerType::get(
            &ctx,
            32,
            pliron::builtin::types::Signedness::Signless,
        )
        .into();
        let block = BasicBlock::new(&mut ctx, None, vec![i32_ty]);
        let (alloca_op, slot) = ValueMap::emit_alloca(&mut ctx, i32_ty, block, None);
        let fragment = DebugFragmentVariableInfo {
            variable: DebugLocalVariableInfo {
                name: "pair".to_string(),
                argument_index: None,
                ty: DebugLocalTypeKind::Basic {
                    name: "u64".to_string(),
                    size_bits: 64,
                    encoding: "DW_ATE_unsigned",
                },
            },
            fragment: DebugFragment {
                offset_bits: 0,
                size_bits: 32,
            },
            source_scope: None,
            declaration: None,
        };
        llvm_export::ops::set_debug_fragment_variables(
            &mut ctx,
            alloca_op,
            std::slice::from_ref(&fragment),
        );

        let value = block.deref(&ctx).get_argument(0);
        let store = Operation::new(
            &mut ctx,
            MirStoreOp::get_concrete_op_info(),
            vec![],
            vec![slot, value],
            vec![],
            0,
        );
        store.insert_after(&ctx, alloca_op);

        materialize_full_debug_fragment_values(&mut ctx, &[block]);

        assert!(llvm_export::ops::debug_fragment_variables(&ctx, alloca_op).is_empty());
        let dbg_values: Vec<_> = block
            .deref(&ctx)
            .iter(&ctx)
            .filter_map(|op| Operation::get_op::<MirDbgValueOp>(op, &ctx))
            .collect();
        assert_eq!(dbg_values.len(), 1);
        assert_eq!(dbg_values[0].value(&ctx), value);
        assert_eq!(
            llvm_export::ops::debug_fragment_variables(&ctx, dbg_values[0].get_operation()),
            vec![fragment]
        );
    }

    /// Exercise `debug_fragment` against composite `VarDebugInfo` produced by
    /// rustc's scalar-replacement pass instead of constructing synthetic types.
    ///
    /// The fixture deliberately creates an aggregate local and enables MIR
    /// optimization so SROA rewrites its debug binding into field fragments.
    /// The test then checks that every supported fragment matches rustc's real
    /// layout and that a non-field projection fails closed.
    #[test]
    fn scalar_replacement_debug_fragments_follow_rustc_layout_and_fail_closed() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_fragment_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("fragment_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub fn scalarized_pair(a: u32, b: u64) -> u64 {
    let pair = (a, b);
    pair.1.wrapping_add(pair.0 as u64)
}
"#,
        )
        .unwrap();

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot_output = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .expect("query rustc sysroot");
        assert!(sysroot_output.status.success(), "rustc --print sysroot");
        let sysroot = String::from_utf8(sysroot_output.stdout)
            .expect("sysroot path is UTF-8")
            .trim()
            .to_string();

        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=fragment_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Cdebuginfo=2".to_string(),
            "-Zmir-opt-level=3".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let result = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let mut checked = 0usize;
                    let mut rejected_non_field = false;

                    for body in rustc_public::all_local_items()
                        .into_iter()
                        .filter_map(|item| item.body())
                    {
                        for info in &body.var_debug_info {
                            if info.name != "pair" {
                                continue;
                            }
                            let Some(composite) = &info.composite else {
                                continue;
                            };
                            let [mir::ProjectionElem::Field(field_idx, field_ty)] =
                                composite.projection.as_slice()
                            else {
                                continue;
                            };

                            let fragment = debug_fragment(composite)
                                .expect("SROA field fragment should be supported");
                            let whole_layout = composite.ty.layout().expect("whole layout").shape();
                            let rustc_public::abi::FieldsShape::Arbitrary { offsets } =
                                &whole_layout.fields
                            else {
                                panic!("tuple fragment must use arbitrary field offsets");
                            };
                            let expected_offset_bits = offsets[*field_idx].bytes() as u64 * 8;
                            let expected_size_bits = field_ty
                                .layout()
                                .expect("field layout")
                                .shape()
                                .size
                                .bytes()
                                as u64
                                * 8;

                            assert_eq!(fragment.offset_bits, expected_offset_bits);
                            assert_eq!(fragment.size_bits, expected_size_bits);
                            checked += 1;

                            let mut invalid = composite.clone();
                            invalid.projection[0] = mir::ProjectionElem::Deref;
                            rejected_non_field |= debug_fragment(&invalid).is_none();
                        }
                    }

                    std::ops::ControlFlow::<(), _>::Continue((checked, rejected_non_field))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert!(
            result.0 >= 2,
            "fixture should produce at least two SROA debug fragments, got {}",
            result.0
        );
        assert!(
            result.1,
            "debug_fragment must reject non-field composite projections"
        );
    }

    /// Optimized closure debug info can mix local-backed and constant-backed
    /// fragments after SROA and constant propagation.
    ///
    /// The literal `u64` capture is intentionally constant while the `u32`
    /// capture depends on an argument. Full debug groups both direct fields
    /// into one aggregate mirror so NVPTX sees a single source-variable
    /// location without disabling MIR optimization.
    #[test]
    fn optimized_closure_constant_capture_uses_composite_debug_mirror() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_closure_constant_fragment_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("closure_constant_fragment_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub fn closure_constant_fragment(seed: u32) -> u32 {
    let captured_u32 = seed + 10;
    let captured_u64 = 0x1_0000_0020u64;
    let closure = move |x: u32| x + captured_u32 + captured_u64 as u32;
    closure(5)
}
"#,
        )
        .unwrap();

        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot_output = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .expect("query rustc sysroot");
        assert!(sysroot_output.status.success(), "rustc --print sysroot");
        let sysroot = String::from_utf8(sysroot_output.stdout)
            .expect("sysroot path is UTF-8")
            .trim()
            .to_string();

        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=closure_constant_fragment_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Cdebuginfo=2".to_string(),
            "-Copt-level=3".to_string(),
            "-Zmir-opt-level=3".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        let result = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let mut rustc_constant_fragments = 0usize;
                    let mut mirror_constant_fragments = 0usize;
                    let mut mirror_place_fragments = 0usize;

                    for body in rustc_public::all_local_items()
                        .into_iter()
                        .filter_map(|item| item.body())
                    {
                        let has_closure_binding = body
                            .var_debug_info
                            .iter()
                            .any(|info| info.name == "closure" && info.composite.is_some());
                        if !has_closure_binding {
                            continue;
                        }

                        rustc_constant_fragments += body
                            .var_debug_info
                            .iter()
                            .filter(|info| {
                                info.name == "closure"
                                    && info.composite.is_some()
                                    && matches!(&info.value, mir::VarDebugInfoContents::Const(_))
                            })
                            .count();

                        let mut ctx = Context::new();
                        let collected = collect_debug_locals(&mut ctx, &body);
                        for mirror in collected
                            .composite_mirrors
                            .iter()
                            .filter(|mirror| mirror.variable.name == "closure")
                        {
                            mirror_constant_fragments += mirror
                                .fragments
                                .iter()
                                .filter(|fragment| {
                                    matches!(
                                        &fragment.value,
                                        CompositeMirrorFragmentValue::Constant { .. }
                                    )
                                })
                                .count();
                            mirror_place_fragments += mirror
                                .fragments
                                .iter()
                                .filter(|fragment| {
                                    matches!(
                                        &fragment.value,
                                        CompositeMirrorFragmentValue::Place { .. }
                                    )
                                })
                                .count();
                        }
                    }

                    std::ops::ControlFlow::<(), _>::Continue((
                        rustc_constant_fragments,
                        mirror_constant_fragments,
                        mirror_place_fragments,
                    ))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        assert!(
            result.0 >= 1,
            "optimized closure fixture should produce at least one constant composite fragment"
        );
        assert_eq!(
            result.1, result.0,
            "every rustc constant direct-field fragment should enter the composite debug mirror"
        );
        assert!(
            result.2 >= 1,
            "optimized closure fixture should mirror at least one local-backed fragment"
        );
    }

    /// Closure environments must be described as composite debug types with
    /// member offsets taken from rustc's real layout, not declaration order.
    ///
    /// `debug_type_for_ty` needs a live compiler session (closure types and
    /// layouts only exist inside one), so this test drives the pinned rustc
    /// in-process on a small fixture via `rustc_public::run!`, extracts the
    /// closure-typed local, and asserts on the returned plain data outside
    /// the session. This fixture intentionally uses `-Zmir-opt-level=0` so the
    /// closure local survives long enough to isolate composite-type layout. Full
    /// device builds keep normal MIR optimization; the scalarized-fragment path
    /// is covered by `scalar_replacement_debug_fragments_follow_rustc_layout_and_fail_closed`.
    ///
    /// The `u32`-before-`u64` capture order is deliberate: rustc's layout
    /// sorts closure fields by descending alignment, placing the `u64` at
    /// offset 0 and the `u32` at offset 8. Sequential declaration-order
    /// offsets would put `capture_0` at 0, so this fails loudly if the
    /// composite type ever stops using the layout's field offsets.
    #[test]
    fn closure_environment_debug_type_uses_layout_offsets() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time before unix epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "cuda_oxide_closure_debug_{}_{}",
            std::process::id(),
            unique
        ));
        std::fs::create_dir_all(&root).unwrap();
        let fixture = root.join("closure_debug_fixture.rs");
        std::fs::write(
            &fixture,
            r#"
pub fn closure_host(a: u32, b: u64) -> u32 {
    let add = move |x: u32| x + a + (b as u32);
    add(1)
}
"#,
        )
        .unwrap();

        // The rustup shim resolves the same pinned toolchain this test binary
        // was built with, so the in-process driver and the sysroot agree.
        let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
        let sysroot_output = std::process::Command::new(rustc)
            .args(["--print", "sysroot"])
            .output()
            .expect("query rustc sysroot");
        assert!(sysroot_output.status.success(), "rustc --print sysroot");
        let sysroot = String::from_utf8(sysroot_output.stdout)
            .expect("sysroot path is UTF-8")
            .trim()
            .to_string();

        let args = vec![
            "rustc".to_string(),
            "--edition=2024".to_string(),
            "--crate-type=rlib".to_string(),
            "--crate-name=closure_debug_fixture".to_string(),
            "--emit=metadata".to_string(),
            "-Zmir-opt-level=0".to_string(),
            format!("--out-dir={}", root.display()),
            format!("--sysroot={sysroot}"),
            fixture.display().to_string(),
        ];

        // rustc needs more stack than the default test-thread allowance.
        let debug_type = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                rustc_public::run!(&args, || {
                    let closure_ty = rustc_public::all_local_items()
                        .into_iter()
                        .filter_map(|item| item.body())
                        .flat_map(|body| body.locals().to_vec())
                        .map(|decl| decl.ty)
                        .find(|ty| matches!(ty.kind(), TyKind::RigidTy(RigidTy::Closure(..))))
                        .expect("fixture must contain a closure-typed local");
                    std::ops::ControlFlow::<(), _>::Continue(debug_type_for_ty(&closure_ty))
                })
            })
            .unwrap()
            .join()
            .unwrap()
            .expect("in-process fixture compilation succeeds");

        std::fs::remove_dir_all(&root).ok();

        let Some(DebugLocalTypeKind::Struct {
            size_bits, members, ..
        }) = debug_type
        else {
            panic!("closure environment must produce a composite debug type, got {debug_type:?}");
        };
        assert_eq!(size_bits, 128, "u64 + u32 environment is 16 bytes");
        assert_eq!(members.len(), 2, "one member per capture");

        assert_eq!(members[0].name, "capture_0");
        assert_eq!(
            members[0].offset_bits, 64,
            "the u32 capture sits after the u64 in rustc's layout"
        );
        match &members[0].ty {
            DebugLocalTypeKind::Basic {
                name, size_bits, ..
            } => {
                assert_eq!(name, "u32");
                assert_eq!(*size_bits, 32);
            }
            other => panic!("capture_0 must be a basic u32, got {other:?}"),
        }

        assert_eq!(members[1].name, "capture_1");
        assert_eq!(
            members[1].offset_bits, 0,
            "the u64 capture is layout-first despite being declared second"
        );
        match &members[1].ty {
            DebugLocalTypeKind::Basic {
                name, size_bits, ..
            } => {
                assert_eq!(name, "u64");
                assert_eq!(*size_bits, 64);
            }
            other => panic!("capture_1 must be a basic u64, got {other:?}"),
        }
    }
}
