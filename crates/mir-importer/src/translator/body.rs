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
use super::types;
use crate::error::{TranslationErr, TranslationResult};
use crate::translator::location::span_to_location;
use crate::translator::values::{self, SlotAddrSpaceMap, ValueMap};
use dialect_mir::ops::MirFuncOp;
use dialect_mir::types::address_space;
use llvm_export::export::DebugKind;
use llvm_export::ops::{
    DebugLocalTypeKind, DebugLocalVariableInfo, DebugSourceScopeMap, DebugTypeMember,
};
use pliron::basic_block::BasicBlock;
use pliron::builtin::op_interfaces::SymbolOpInterface;
use pliron::context::{Context, Ptr};
use pliron::identifier::{Identifier, Legaliser};
use pliron::input_err_noloc;
use pliron::location::Located;
use pliron::op::Op;
use pliron::operation::Operation;

// Re-export rustc_public types for convenience
use rustc_hash::FxHashMap;
use rustc_public::CrateDef;
use rustc_public::mir;
use rustc_public::mir::mono;
use rustc_public::ty::{ConstantKind, FloatTy, IntTy, RigidTy, Ty, TyKind, UintTy};

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

/// Scans MIR for `__cluster_config::<X, Y, Z>()` marker and extracts cluster dimensions.
///
/// The `#[cluster(x,y,z)]` macro injects this call at the start of the kernel.
/// We scan the MIR to find it and extract the const generic parameters.
///
/// Returns `Some(ClusterDims)` if found, `None` otherwise.
fn detect_cluster_config(body: &mir::Body) -> Option<ClusterDims> {
    use rustc_public::ty::TyConstKind;

    for block in &body.blocks {
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
fn detect_launch_bounds_config(body: &mir::Body) -> Option<LaunchBounds> {
    use rustc_public::ty::TyConstKind;

    for block in &body.blocks {
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
        if fn_name != "__launch_bounds_config" && !fn_name.ends_with("::__launch_bounds_config") {
            continue;
        }

        // Extract const generic args (MAX_THREADS, MIN_BLOCKS)
        let mut values = [0u32, 0u32];
        for (i, arg) in args.0.iter().take(2).enumerate() {
            let rustc_public::ty::GenericArgKind::Const(c) = arg else {
                continue;
            };
            values[i] = match c.kind() {
                TyConstKind::Value(_, alloc) => alloc.read_uint().ok().map(|v| v as u32),
                _ => c.eval_target_usize().ok().map(|v| v as u32),
            }
            .unwrap_or(values[i]);
        }

        return Some(LaunchBounds {
            max_threads: values[0],
            min_blocks: values[1],
        });
    }
    None
}

/// Return the non-unwind successors of a terminator.
///
/// [`mir::Terminator::successors`] includes unwind cleanup blocks alongside
/// "normal" control-flow targets. The CUDA toolchain does not support stack
/// unwinding (hardware could, but `nvcc`/`ptxas` never wire it up), so the
/// translator treats unwind cleanups as dead code. This helper strips them
/// out so the worklist only visits blocks that matter on GPU.
fn non_unwind_successors(kind: &mir::TerminatorKind) -> Vec<usize> {
    use mir::TerminatorKind::*;
    match kind {
        Goto { target } => vec![*target],
        SwitchInt { targets, .. } => targets.all_targets(),
        Return | Resume | Abort | Unreachable => vec![],
        Drop { target, .. } | Assert { target, .. } => vec![*target],
        Call { target, .. } => target.map(|t| vec![t]).unwrap_or_default(),
        InlineAsm { destination, .. } => destination.map(|t| vec![t]).unwrap_or_default(),
    }
}

/// BFS from the entry block (index 0) following non-unwind successors.
///
/// The result is a sorted set of reachable-on-GPU block indices; unwind-only
/// cleanup blocks end up outside this set and are filled in with
/// `mir.unreachable` by [`translate_body`] so pliron verification still
/// passes.
fn compute_reachable_blocks(body: &mir::Body) -> std::collections::BTreeSet<usize> {
    let mut reachable = std::collections::BTreeSet::new();
    let mut frontier: Vec<usize> = vec![0];
    reachable.insert(0);
    while let Some(idx) = frontier.pop() {
        let successors = non_unwind_successors(&body.blocks[idx].terminator.kind);
        for succ in successors {
            if reachable.insert(succ) {
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

/// Build the first full-debug variable map.
///
/// This stage only supports simple whole-local bindings:
///
/// ```text
/// debug name => _3
/// ```
///
/// Fragments/projections need `DIExpression(DW_OP_LLVM_fragment, ...)` and more
/// value-location tracking, so they are intentionally skipped until the basic
/// local/argument path is solid.
fn collect_debug_locals(
    ctx: &mut Context,
    body: &mir::Body,
) -> FxHashMap<mir::Local, LocalDebugInfo> {
    let mut locals = FxHashMap::default();

    for info in &body.var_debug_info {
        if info.composite.is_some() {
            continue;
        }

        let Some(local) = info.local() else {
            continue;
        };
        let local_idx: usize = local;
        if local_idx == 0 {
            continue;
        }

        let Some(decl) = body.local_decl(local) else {
            continue;
        };
        let Some(ty) = debug_type_for_ty(&decl.ty) else {
            continue;
        };

        let name = info.name.to_string();
        if name.is_empty() {
            continue;
        }

        locals.entry(local).or_insert_with(|| LocalDebugInfo {
            variable: DebugLocalVariableInfo {
                name,
                argument_index: info.argument_index,
                ty,
            },
            loc: span_to_location(ctx, info.source_info.span),
            source_scope: info.source_info.scope,
        });
    }

    locals
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
            // Only plain structs (one variant) are described as composites here;
            // enums and unions need DWARF variant parts (deferred).
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
) -> Option<Ptr<Operation>> {
    let mut prev_op: Option<Ptr<Operation>> = None;
    let debug_locals = if debug_kind.variables_enabled() {
        collect_debug_locals(ctx, body)
    } else {
        FxHashMap::default()
    };

    // Pre-scan the body once: for each local whose translated slot type is a
    // pointer, infer the address space from the *writes* into it rather than
    // trusting Rust's declared type (which loses addrspace info for
    // references / raw pointers).
    let slot_addr_spaces = SlotAddrSpaceMap::analyze(body);

    for local_idx in 0..body.locals().len() {
        let local = mir::Local::from(local_idx);
        let local_ty = &body.locals()[local].ty;
        if types::is_rust_type_zst(local_ty) {
            continue;
        }
        let mir_ty = match types::translate_type(ctx, local_ty) {
            Ok(t) => t,
            Err(_) => continue,
        };

        // Override the Rust-declared addrspace with the inferred one for
        // pointer slots. Non-pointer slots are untouched by
        // `align_pointer_addr_space`.
        let rust_declared =
            values::pointer_addr_space(ctx, mir_ty).unwrap_or(address_space::GENERIC);
        let target = slot_addr_spaces.effective(local, rust_declared);
        let mir_ty = values::align_pointer_addr_space(ctx, mir_ty, target);

        let (op, slot) = ValueMap::emit_alloca(ctx, mir_ty, entry_block, prev_op);
        if let Some(info) = debug_locals.get(&local) {
            llvm_export::ops::set_debug_local_variable(ctx, op, info.variable.clone());
            if debug_source_scopes
                .is_some_and(|map| map.scopes.iter().any(|scope| scope.id == info.source_scope))
            {
                llvm_export::ops::set_debug_local_source_scope(ctx, op, info.source_scope);
            }
            op.deref_mut(ctx).set_loc(info.loc.clone());
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
/// * `is_kernel` - Add `gpu_kernel` attribute for kernel entry points
/// * `is_inline_always` - Add `alwaysinline` attribute (non-kernel functions
///   marked `#[inline(always)]` in rustc)
/// * `override_name` - Custom export name (defaults to instance name)
pub fn translate_body(
    ctx: &mut Context,
    body: &mir::Body,
    instance: &mono::Instance,
    is_kernel: bool,
    is_inline_always: bool,
    override_name: Option<&str>,
    legaliser: &mut Legaliser,
    debug_kind: DebugKind,
    debug_source_scopes: Option<&DebugSourceScopeMap>,
) -> TranslationResult<Ptr<Operation>> {
    // Create a value map to track MIR locals -> pliron IR values
    let num_locals = body.locals().len();
    let mut value_map = ValueMap::new(num_locals);

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
        if let Some(cluster_dims) = detect_cluster_config(body) {
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
        if let Some(launch_bounds) = detect_launch_bounds_config(body) {
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
    let entry_last_op = emit_entry_allocas(
        ctx,
        body,
        block_map[0],
        num_args,
        &mut value_map,
        debug_kind,
        debug_source_scopes,
    );

    // -------------------------------------------------------------------------
    // PHASE 2: Translate reachable blocks
    // -------------------------------------------------------------------------
    //
    // Every local flows through its stack slot, so blocks have no inter-block
    // ordering dependency and can be translated in a single index-order pass.
    // Unwind-only cleanup blocks are skipped here (see
    // [`non_unwind_successors`]) and patched with `mir.unreachable` below.
    let reachable: std::collections::BTreeSet<usize> = compute_reachable_blocks(body);

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

    Ok(op_ptr)
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
}
