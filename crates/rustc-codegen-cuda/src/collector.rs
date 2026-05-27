/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! # Device Function Collector
//!
//! This module identifies all functions that must be compiled for the GPU, starting
//! from kernel entry points and transitively collecting all reachable callees.
//!
//! ## How It Works
//!
//! The collector performs a breadth-first traversal of the MIR call graph:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────────────────┐
//! │                         DEVICE FUNCTION COLLECTION                              │
//! │                                                                                 │
//! │   Input: Codegen Units (CGUs) from rustc                                        │
//! │          Each CGU contains monomorphized function instances                     │
//! │                                                                                 │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 1: Find Kernel Entry Points                                       │   │
//! │   │                                                                         │   │
//! │   │  Scan all CGUs for functions whose names contain the reserved           │   │
//! │   │  KERNEL_PREFIX from `reserved-oxide-symbols` (the #[kernel] macro       │   │
//! │   │  renames `fn foo` into the hash-suffixed `cuda_oxide_*` namespace).     │   │
//! │   │                                                                         │   │
//! │   │  Example:                                                               │   │
//! │   │    #[kernel]                                                            │   │
//! │   │    fn add_one(data: *mut i32, len: usize) { ... }                       │   │
//! │   │                                                                         │   │
//! │   │    Becomes: cuda_oxide_kernel_<hash>_add_one in MIR                     │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  STEP 2: Walk Call Graph (Worklist Algorithm)                           │   │
//! │   │                                                                         │   │
//! │   │  worklist = [kernel1, kernel2, ...]                                     │   │
//! │   │  seen = {}                                                              │   │
//! │   │  result = []                                                            │   │
//! │   │                                                                         │   │
//! │   │  while worklist not empty:                                              │   │
//! │   │      func = worklist.pop()                                              │   │
//! │   │      mir = tcx.instance_mir(func)  ◄─── Gets OPTIMIZED MIR              │   │
//! │   │                                                                         │   │
//! │   │      for terminator in mir.basic_blocks:                                │   │
//! │   │          if terminator is Call:                                         │   │
//! │   │              callee = resolve_callee(terminator)                        │   │
//! │   │              if should_collect(callee) and callee not in seen:          │   │
//! │   │                  worklist.push(callee)                                  │   │
//! │   │                  seen.insert(callee)                                    │   │
//! │   │                                                                         │   │
//! │   │      result.push(func)                                                  │   │
//! │   │                                                                         │   │
//! │   │  return result                                                          │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                              │                                                  │
//! │                              ▼                                                  │
//! │   ┌─────────────────────────────────────────────────────────────────────────┐   │
//! │   │  Output: Vec<CollectedFunction>                                         │   │
//! │   │                                                                         │   │
//! │   │  Each contains:                                                         │   │
//! │   │    - instance: The monomorphized Instance<'tcx>                         │   │
//! │   │    - is_kernel: true for entry points, false for callees                │   │
//! │   │    - export_name: Name to use in PTX                                    │   │
//! │   └─────────────────────────────────────────────────────────────────────────┘   │
//! │                                                                                 │
//! └─────────────────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Which Functions Are Collected
//!
//! We collect functions from these crates:
//!
//! | Crate                    | What's Collected                               | What's Filtered                          |
//! |--------------------------|------------------------------------------------|------------------------------------------|
//! | Local crate              | Everything reachable from kernels              | —                                        |
//! | External crates          | Kernels (`cuda_oxide_kernel_<hash>_*`)         | —                                        |
//! | `cuda_device`            | Non-intrinsic functions                        | Intrinsic stubs (just `unreachable!()`)  |
//! | `core`                   | Iterators, Option, etc.                        | `fmt::*`, `panicking::*`                 |
//! | `alloc`                  | Vec, Box, String (if GPU allocator configured) | —                                        |
//! | Other `no_std` crates    | All reachable functions                        | —                                        |
//!
//! ## Cross-Crate Kernel Support
//!
//! Library crates can export generic kernels that get monomorphized when used:
//!
//! ```rust,ignore
//! // my_cuda_lib/src/lib.rs
//! #[kernel]
//! pub fn reduce<T: Add>(data: &[T], out: &mut T) { ... }
//!
//! // my_app/src/main.rs
//! use my_cuda_lib::reduce;
//! cuda_launch! { kernel: reduce::<f32>, ... }  // PTX generated here!
//! ```
//!
//! Functions from `std` are FORBIDDEN because they require OS/threads/IO.
//!
//! ## MIR Access
//!
//! We access MIR via `tcx.instance_mir(instance.def)`, which returns **optimized MIR**.
//! This is the same MIR that would go to LLVM for native compilation. The optimization
//! level depends on the `-C opt-level` flag passed to rustc.
//!
//! ## Export Names and FQDN Alignment
//!
//! Export names must match what the MIR translator (`extract_func_info` in
//! `terminator/mod.rs`) produces for call targets. Both sides use fully qualified
//! domain names (FQDNs), which the lowering layer converts `::` to `__` for
//! valid LLVM/PTX identifiers.
//!
//! The collector uses [`DeviceCollector::fqdn()`] to produce FQDNs matching
//! `CrateDef::name()` on the `rustc_public` side. For local items, this
//! prepends the crate name to `def_path_str()`.
//!
//! For generic/complex names like `ptr::add::<i32>`, we use the mangled
//! symbol name (e.g., `_RNvMNtNtCs...`) because `<`, `>` are not valid
//! PTX identifiers. The MIR translator uses the same mangling for generic calls.
//!
//! Kernel export names are separate — they use `compute_kernel_export_name`
//! with human-readable base names derived from the `#[kernel]` macro.
//!
//! This naming strategy will be replaced by pliron's `Legaliser` when
//! the framework is upgraded (see metal-oxide for reference).

use rustc_hir::def_id::{DefId, LOCAL_CRATE};
use rustc_middle::mir::TerminatorKind;
use rustc_middle::mir::mono::{CodegenUnit, MonoItem};
use rustc_middle::ty::{Instance, InstanceKind, Ty, TyCtxt, TyKind, TypeVisitableExt, TypingEnv};
use std::collections::{HashSet, VecDeque};

/// Result of checking if a function should be collected for device compilation.
#[derive(Debug)]
enum CollectDecision {
    /// Collect this function for device compilation.
    Collect,
    /// Skip this function intentionally (e.g., core::fmt::*, core::panicking::*).
    /// These are filtered out because they can't compile to PTX, but calling them
    /// is expected (panic paths, debug assertions) and will be handled by panic=abort.
    SkipIntentional,
    /// Error: function is from a forbidden crate (std, alloc, etc.).
    /// Device code cannot call these - this is a user error.
    Forbidden { crate_name: String, fn_path: String },
}

// The prefix constants and substring/extractor helpers used below
// (`KERNEL_PREFIX`, `is_kernel_symbol`, `kernel_base_name`, etc.) live in
// the workspace-internal `reserved-oxide-symbols` crate. That crate is the
// single source of truth for the cuda_oxide_* naming contract; see its
// crate-level docs for the layered API and the hash-suffix rationale.
//
// Each prefix ends with the magic suffix `246e25db_`, which makes a
// substring like "cuda_oxide_kernel_" — without the hash — never falsely
// match. The mutual-exclusion guarantee between `DEVICE_PREFIX` and
// `DEVICE_EXTERN_PREFIX` means we no longer need the historical
// "test extern first" ordering dance that lived here previously.
use reserved_oxide_symbols::{
    device_extern_base_name, is_device_extern_symbol, is_device_symbol, is_kernel_symbol,
    kernel_base_name,
};

/// Sanitize a symbol name for use as a PTX identifier.
///
/// PTX identifiers must match `[a-zA-Z_][a-zA-Z0-9_]*`. This function:
/// - Replaces `$` with `_` (legacy mangling uses `$LT$`, `$GT$`, `$u20$`, etc.)
/// - Replaces `.` with `_` (legacy mangling uses `..` for `::`)
///
/// This must be kept in sync with `mir-importer/src/translator/terminator/mod.rs`
/// which sanitizes call target names the same way.
pub fn sanitize_ptx_name(name: &str) -> String {
    name.replace(['$', '.'], "_")
}

/// Compute the export name for a kernel.
///
/// Naming scheme:
/// - Non-generic kernel (no type args)  -> `base_name`
/// - Generic kernel with N type args    -> `base_name + "_TID_" + hex32`
///
/// where `hex32` is the lowercase hex form of
/// `tcx.type_id_hash(tuple_ty).as_u128()` and `tuple_ty` is
/// `Ty::new_tup(tcx, &[arg0, arg1, ...])`. We hash the tuple — not each
/// arg separately — so the on-wire name stays at a fixed length
/// (`base.len() + 37`) regardless of generic arity. PTX identifiers can
/// be ~1024 chars, but the name shows up many times per kernel
/// (`<name>_param_N`) and a per-arg layout would grow linearly with the
/// number of generic parameters.
///
/// The host computes the same value via
/// `cuda_host::type_id_u128::<(T0, T1, ...)>()`. Both sides go through
/// `erase_and_anonymize_regions` + the same stable-hash pipeline, so the
/// 1-tuple `(T,)` from the macro matches `Ty::new_tup(tcx, &[T])` here.
///
/// The scheme is uniform — closures, named types, integers, references
/// — all funnel through one path. That intentionally collapses the
/// older closure-special-case (`_L<line>C<col>`) and the older named-
/// type case (`_Debug-formatted_name`) into the same shape, so closure-
/// generic kernels (`map<T, F: Fn(T) -> T + Copy>`) can finally be
/// launched through the typed `module.<kernel>(...)` API. The host-side
/// `GenericCudaKernel::ptx_name` impl emitted by `#[kernel]` /
/// `#[cuda_module]` produces the exact same string from the type
/// parameters it sees at the call site.
fn compute_kernel_export_name<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: Instance<'tcx>,
    base_name: &str,
) -> String {
    let type_args: Vec<Ty<'tcx>> = instance
        .args
        .iter()
        .filter_map(|arg| arg.as_type())
        .collect();

    if type_args.is_empty() {
        return base_name.to_string();
    }

    let tuple_ty = Ty::new_tup(tcx, &type_args);
    let hash = tcx.type_id_hash(tuple_ty).as_u128();
    format!("{}_TID_{:032x}", base_name, hash)
}

/// A function collected for GPU compilation.
///
/// This struct captures everything needed to compile a function to PTX:
/// - The monomorphized instance (with concrete generic arguments)
/// - Whether it's a kernel entry point or a device helper
/// - The name to export in PTX
#[derive(Debug, Clone)]
pub struct CollectedFunction<'tcx> {
    /// The fully monomorphized function instance.
    ///
    /// For generic functions like `add::<f32>`, this contains the concrete
    /// type substitutions. We use this to get the MIR body with all types resolved.
    pub instance: Instance<'tcx>,

    /// True if this is a GPU kernel entry point.
    ///
    /// Kernels are marked with `.entry` in PTX and can be launched from the host.
    /// Non-kernel functions are marked with `.func` and can only be called from device code.
    pub is_kernel: bool,

    /// The name to export in PTX.
    ///
    /// For kernels: the user-visible name (e.g., `add_one`)
    /// For generics: the mangled symbol name (e.g., `_RNvMNtNtCs...`)
    pub export_name: String,
}

/// An external device function declaration (for linking with external LTOIR).
///
/// Unlike `CollectedFunction`, these have no MIR body - they're just declarations
/// that will be emitted as LLVM `declare` statements for nvJitLink to resolve.
#[derive(Debug, Clone)]
pub struct DeviceExternDecl {
    /// The DefId of the extern function declaration
    pub def_id: DefId,

    /// The export name (the original function name, e.g., "cub_block_reduce_sum")
    pub export_name: String,

    /// NVVM attributes extracted from the declaration
    pub attrs: DeviceExternAttrs,
}

/// NVVM attributes for device extern declarations.
///
/// NOTE: These attributes are currently **not emitted** to the LLVM IR output.
/// When linking LTOIR via nvJitLink, the external library's LTOIR already contains
/// proper attributes (convergent, nounwind, memory, etc.) on the function DEFINITIONS.
/// nvJitLink uses the definition's attributes during LTO, making attributes on our
/// declarations redundant.
///
/// The `#[convergent]`, `#[pure]`, and `#[readonly]` Rust attributes are still parsed
/// but their values are not used in code generation. This struct is retained for
/// potential future use or debugging.
#[derive(Debug, Clone, Default)]
pub struct DeviceExternAttrs {
    /// Function is convergent (all threads must execute together).
    /// NOTE: Not currently emitted - external LTOIR has proper convergent attrs.
    pub is_convergent: bool,

    /// Function is pure (no side effects, result depends only on inputs).
    /// NOTE: Not currently emitted - external LTOIR has proper memory attrs.
    pub is_pure: bool,

    /// Function is read-only (only reads memory, doesn't write).
    /// NOTE: Not currently emitted - external LTOIR has proper memory attrs.
    pub is_readonly: bool,
}

/// Result of the collection process.
///
/// Contains both compiled functions (with MIR bodies) and external device
/// declarations (for FFI with external LTOIR).
#[derive(Debug)]
pub struct CollectionResult<'tcx> {
    /// Functions to compile (kernels and device helpers with MIR bodies).
    pub functions: Vec<CollectedFunction<'tcx>>,

    /// External device function declarations (no MIR, emit as `declare`).
    pub device_externs: Vec<DeviceExternDecl>,
}

/// Counts kernel functions across all codegen units.
///
/// This is a quick scan to determine if device compilation is needed.
/// Returns 0 if no kernels are found, allowing the backend to skip device codegen entirely.
///
/// Note: Only counts fully monomorphized kernels. Generic kernel definitions
/// (like `scale<T>`) are skipped - only concrete instantiations count.
pub fn count_kernels_in_cgus<'tcx>(tcx: TyCtxt<'tcx>, cgus: &[CodegenUnit<'tcx>]) -> usize {
    let mut count = 0;
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_kernel_function(tcx, instance.def_id())
                && is_fully_monomorphized(tcx, *instance)
            {
                count += 1;
            }
        }
    }
    count
}

/// Counts standalone device function definitions across all codegen units.
///
/// Returns 0 if no standalone device functions are found.
/// Used alongside `count_kernels_in_cgus` to determine if device compilation is needed.
pub fn count_device_fns_in_cgus<'tcx>(tcx: TyCtxt<'tcx>, cgus: &[CodegenUnit<'tcx>]) -> usize {
    let mut count = 0;
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_device_function(tcx, instance.def_id())
                && is_fully_monomorphized(tcx, *instance)
            {
                count += 1;
            }
        }
    }
    count
}

/// Checks if a function is a kernel entry point.
///
/// Detection is based on the `KERNEL_PREFIX` substring (currently
/// `cuda_oxide_kernel_246e25db_`) which the `#[kernel]` macro adds to
/// renamed functions:
///
/// ```text
/// User writes:        Macro expands to:
/// ┌─────────────────┐  ┌────────────────────────────────────────────────┐
/// │ #[kernel]       │  │ #[no_mangle]                                   │
/// │ fn add_one(...) │ ⇒│ pub fn cuda_oxide_kernel_246e25db_add_one(...) │
/// └─────────────────┘  └────────────────────────────────────────────────┘
/// ```
pub fn is_kernel_function(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    is_kernel_symbol(&tcx.def_path_str(def_id))
}

/// Checks if a function is a standalone device function definition.
///
/// Detection is based on the `DEVICE_PREFIX` substring added by the
/// `#[device]` macro on `fn` items. The mutual-exclusion property
/// documented in `reserved-oxide-symbols` means we don't need an explicit
/// exclusion of device-extern symbols here — `is_device_symbol` handles it.
pub fn is_device_function(tcx: TyCtxt<'_>, def_id: DefId) -> bool {
    is_device_symbol(&tcx.def_path_str(def_id))
}

/// Checks if an Instance is fully monomorphized (no unresolved type parameters).
///
/// For generic kernels like `scale<T>`, the CGU may contain both:
/// - The generic definition (with T as a type parameter)
/// - Concrete instantiations (with T = f32, T = i32, etc.)
///
/// We only want to process concrete instantiations since we can't generate
/// PTX for generic code - the device compiler needs concrete types.
///
/// Returns false if any substitution argument is still a type parameter.
pub fn is_fully_monomorphized<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> bool {
    let generics = tcx.generics_of(instance.def_id());

    // First check: does the Instance itself have any unresolved type parameters?
    // The `args` field contains the substitutions for this instance.
    // For scale::<f32>, args would be [f32]
    // For scale<T> (generic), args would be [T/#0] (a type parameter)
    for arg in instance.args.iter() {
        if let Some(ty) = arg.as_type()
            && ty.has_param()
        {
            return false;
        }
    }

    // Second check: does the def itself have generics that need substitution?
    // Even if args is empty, the function might be generic but not properly instantiated.
    if generics.count() > 0 && instance.args.is_empty() {
        return false;
    }

    true
}

/// Collects all device-reachable functions starting from kernel entry points.
///
/// This is the main entry point for device function collection. It:
///
/// 1. Finds all kernel entry points in the CGUs
/// 2. Walks the call graph from each kernel
/// 3. Returns all functions that need to be compiled to PTX
///
/// ## Parameters
///
/// - `tcx`: The type context containing all MIR bodies
/// - `cgus`: Codegen units from `tcx.collect_and_partition_mono_items()`
/// - `verbose`: If true, prints collection progress to stderr
///
/// ## Returns
///
/// A `CollectionResult` containing:
/// - `functions`: Collected functions with MIR bodies (kernels first, then callees)
/// - `device_externs`: External device function declarations (for FFI with external LTOIR)
pub fn collect_device_functions<'tcx>(
    tcx: TyCtxt<'tcx>,
    cgus: &[CodegenUnit<'tcx>],
    verbose: bool,
) -> CollectionResult<'tcx> {
    let mut collector = DeviceCollector::new(tcx, verbose);

    // Find all kernel entry points
    for cgu in cgus {
        for (item, _data) in cgu.items() {
            if let MonoItem::Fn(instance) = item
                && is_kernel_function(tcx, instance.def_id())
            {
                // Skip closures inside kernels - they are device functions, not kernels.
                // Closures have names like "cuda_oxide_kernel_<hash>_foo::{closure#0}" but
                // only "cuda_oxide_kernel_<hash>_foo" is the actual kernel entry point.
                let name = tcx.def_path_str(instance.def_id());
                if name.contains("{closure") || name.contains("::closure") {
                    if verbose {
                        eprintln!(
                            "[collector] Skipping closure inside kernel (not an entry point): {}",
                            name
                        );
                    }
                    continue;
                }

                // Skip generic (non-monomorphized) instances.
                // For generic kernels like scale<T>, the CGU contains both:
                // - The generic definition (scale<T>) - skip this
                // - Concrete instantiations (scale::<f32>) - process this
                if !is_fully_monomorphized(tcx, *instance) {
                    if verbose {
                        let name = tcx.def_path_str(instance.def_id());
                        eprintln!(
                            "[collector] Skipping non-monomorphized kernel: {} (needs type instantiation)",
                            name
                        );
                    }
                    continue;
                }

                let name = tcx.def_path_str(instance.def_id());
                // Extract the kernel base name by stripping the reserved
                // `cuda_oxide_kernel_<hash>_` prefix. Cross-crate kernels look
                // like `kernel_lib::cuda_oxide_kernel_<hash>_scale`; the
                // helper handles both bare and FQDN forms uniformly.
                let base_name = kernel_base_name(&name)
                    .map(str::to_string)
                    .unwrap_or_else(|| name.rsplit("::").next().unwrap_or(&name).to_string());

                // Compute a unique export name for this kernel monomorphization.
                // Non-generic kernels keep the base name (e.g. "vecadd").
                // Generic kernels (including closure-generic) get
                // "<base>_TID_<hex32>", where <hex32> is the hash of the
                // *tuple* of generic args (constant length regardless of
                // arity). The host-side `ptx_name()` emitted by `#[kernel]`
                // / `#[cuda_module]` computes the same string.
                let export_name = compute_kernel_export_name(tcx, *instance, &base_name);

                if verbose {
                    eprintln!("[collector] Found kernel: {} -> {}", name, export_name);
                }

                collector.add_root(*instance, true, export_name);
            }
        }
    }

    // Find standalone device function roots (Phase 2: device functions without kernels).
    // Only scan when there are NO kernels — when kernels exist, device functions are
    // already collected transitively via the call graph walk.
    if collector.worklist.is_empty() {
        for cgu in cgus {
            for (item, _data) in cgu.items() {
                if let MonoItem::Fn(instance) = item
                    && is_device_function(tcx, instance.def_id())
                    && is_fully_monomorphized(tcx, *instance)
                {
                    let raw_name = tcx.def_path_str(instance.def_id());

                    // Skip closures inside device functions
                    if raw_name.contains("{closure") || raw_name.contains("::closure") {
                        continue;
                    }

                    // Use FQDN so the export name matches what the MIR translator
                    // sees via `CrateDef::name()` on the call side. The lowering
                    // layer converts `::` to `__` on both sides.
                    let name = collector.fqdn(instance.def_id());
                    let export_name = collector.compute_export_name(&name, *instance);

                    if verbose {
                        eprintln!(
                            "[collector] Found standalone device function: {} -> {}",
                            name, export_name
                        );
                    }

                    // Add as a non-kernel root — produces .func (not .entry) in PTX
                    collector.add_root(*instance, false, export_name);
                }
            }
        }
    }

    // Process the worklist to collect all reachable functions
    collector.collect()
}

/// Worklist-based collector for device-reachable functions.
///
/// Uses breadth-first traversal to discover all functions reachable from kernels.
/// This ensures we don't miss any callees, even through deep call chains.
struct DeviceCollector<'tcx> {
    tcx: TyCtxt<'tcx>,
    /// Mangled names of functions already seen (prevents duplicates and infinite loops).
    /// We use mangled names because they uniquely identify each monomorphization,
    /// unlike DefId which is shared across all instantiations of a generic function.
    seen: HashSet<String>,
    /// Export names already used (prevents name conflicts in PTX).
    used_export_names: HashSet<String>,
    /// Functions awaiting processing.
    worklist: VecDeque<CollectedFunction<'tcx>>,
    /// Functions collected so far, in discovery order.
    result: Vec<CollectedFunction<'tcx>>,
    /// External device function declarations collected (for FFI with external LTOIR).
    device_externs: Vec<DeviceExternDecl>,
    /// DefIds of device externs already seen (prevents duplicates).
    seen_device_externs: HashSet<DefId>,
    /// Whether we've already emitted the "`DynamicSharedArray::get`
    /// needs `shared_mem_bytes` set at launch" warning. Fired at most
    /// once per program — the message is procedural advice about the
    /// kernel-launch contract, not anything per-call.
    warned_dynamic_shared_array: bool,
    /// Print progress to stderr.
    verbose: bool,
}

impl<'tcx> DeviceCollector<'tcx> {
    fn new(tcx: TyCtxt<'tcx>, verbose: bool) -> Self {
        Self {
            tcx,
            seen: HashSet::new(),
            used_export_names: HashSet::new(),
            worklist: VecDeque::new(),
            result: Vec::new(),
            device_externs: Vec::new(),
            seen_device_externs: HashSet::new(),
            warned_dynamic_shared_array: false,
            verbose,
        }
    }

    /// Returns the fully qualified domain name (FQDN) for a DefId.
    ///
    /// `def_path_str()` omits the crate name for local items (e.g. returns
    /// `cuda_oxide_device_<hash>_vecadd` instead of
    /// `helper_fn::cuda_oxide_device_<hash>_vecadd`).
    /// This method prepends the local crate name so the result matches what
    /// `CrateDef::name()` returns on the `rustc_public` side, ensuring that
    /// call sites and definitions use identical strings before lowering
    /// converts `::` to `__`.
    fn fqdn(&self, def_id: DefId) -> String {
        let path = self.tcx.def_path_str(def_id);
        if def_id.krate == LOCAL_CRATE {
            format!("{}::{}", self.tcx.crate_name(LOCAL_CRATE), path)
        } else {
            path
        }
    }

    /// Adds a root function (kernel) to start collection from.
    fn add_root(&mut self, instance: Instance<'tcx>, is_kernel: bool, export_name: String) {
        // Use mangled name as the unique key - this distinguishes different
        // monomorphizations of the same generic function (e.g., map<f32, Closure1>
        // vs map<f32, Closure2>)
        let mangled = self.tcx.symbol_name(instance).name.to_string();
        if self.seen.insert(mangled) {
            self.used_export_names.insert(export_name.clone());
            self.worklist.push_back(CollectedFunction {
                instance,
                is_kernel,
                export_name,
            });
        }
    }

    /// Runs collection to completion, returning all discovered functions and extern declarations.
    fn collect(mut self) -> CollectionResult<'tcx> {
        while let Some(func) = self.worklist.pop_front() {
            let def_id = func.instance.def_id();

            // Get MIR body if available
            if self.tcx.is_mir_available(def_id) {
                // Use instance_mir for monomorphized MIR.
                // This returns OPTIMIZED MIR (post -C opt-level passes).
                let mir = self.tcx.instance_mir(func.instance.def);

                if self.verbose {
                    eprintln!(
                        "[collector] Processing {} ({} basic blocks)",
                        func.export_name,
                        mir.basic_blocks.len()
                    );
                }

                // Walk all basic blocks looking for calls.
                // Pass the caller's instance so we can substitute its args into callees.
                for bb_data in mir.basic_blocks.iter() {
                    if let Some(ref terminator) = bb_data.terminator {
                        self.process_terminator(terminator, &func.instance);
                    }
                }
            }

            self.result.push(func);
        }

        CollectionResult {
            functions: self.result,
            device_externs: self.device_externs,
        }
    }

    /// Adds an external device function declaration (for FFI with external LTOIR).
    fn add_device_extern(&mut self, def_id: DefId, full_name: &str) {
        // Skip if already seen
        if !self.seen_device_externs.insert(def_id) {
            return;
        }

        // Extract the original function name (strip the prefix)
        // The #[link_name] attribute on the extern fn has the original name.
        // `device_extern_base_name` returns the part after DEVICE_EXTERN_PREFIX
        // and works for both bare and FQDN forms.
        let export_name = device_extern_base_name(full_name)
            .map(str::to_string)
            .unwrap_or_else(|| full_name.to_string());

        // Extract NVVM attributes from the function's attributes
        let attrs = self.extract_device_extern_attrs(def_id);

        if self.verbose {
            eprintln!(
                "[collector] Found device extern: {} (convergent={}, pure={}, readonly={})",
                export_name, attrs.is_convergent, attrs.is_pure, attrs.is_readonly
            );
        }

        self.device_externs.push(DeviceExternDecl {
            def_id,
            export_name,
            attrs,
        });
    }

    /// Extract NVVM attributes from a device extern function's rustc attributes.
    fn extract_device_extern_attrs(&self, def_id: DefId) -> DeviceExternAttrs {
        use rustc_span::Symbol;
        let mut attrs = DeviceExternAttrs::default();

        let check = |name| {
            self.tcx
                .get_attrs_by_path(def_id, &[Symbol::intern(name)])
                .next()
                .is_some()
        };
        attrs.is_convergent = check("convergent");
        attrs.is_pure = check("pure");
        attrs.is_readonly = check("readonly");

        attrs
    }

    /// Process a terminator to find function calls.
    ///
    /// `caller` is the instance of the function containing this terminator.
    /// We use its args to substitute into callee args when the caller is generic.
    fn process_terminator(
        &mut self,
        terminator: &rustc_middle::mir::Terminator<'tcx>,
        caller: &Instance<'tcx>,
    ) {
        if let TerminatorKind::Call { func, .. } = &terminator.kind {
            self.process_call_operand(func, caller);
        }
    }

    /// Process a call operand to extract and add the callee.
    ///
    /// This is where we enforce the `no_std` requirement for device code.
    /// If the call target is from a forbidden crate (std, alloc, etc.),
    /// we panic with a clear error message.
    ///
    /// `caller` is the instance of the function containing this call, used to
    /// substitute its generic args into the callee's args when needed.
    fn process_call_operand(
        &mut self,
        func: &rustc_middle::mir::Operand<'tcx>,
        caller: &Instance<'tcx>,
    ) {
        use rustc_middle::mir::Operand;
        use rustc_middle::ty::EarlyBinder;

        let Operand::Constant(const_op) = func else {
            return;
        };

        let ty = const_op.const_.ty();
        let TyKind::FnDef(def_id, args) = ty.kind() else {
            return;
        };
        let fn_path = self.tcx.def_path_str(*def_id);
        if fn_path.contains("DynamicSharedArray")
            && (fn_path.contains("::get")
                || fn_path.contains("::get_raw")
                || fn_path.contains("::offset"))
            && !self.warned_dynamic_shared_array
        {
            self.warned_dynamic_shared_array = true;
            self.tcx.sess.dcx().span_warn(
                const_op.span,
                "`DynamicSharedArray` returns CUDA dynamic shared memory; make sure the kernel launch config provides enough `shared_mem_bytes`",
            );
        }

        // CRITICAL: Substitute the caller's args into the callee's args.
        //
        // When walking the MIR of a generic function like
        // `cuda_oxide_kernel_<hash>_scale<T>`, calls to other functions may have
        // generic args like `[T]`. We substitute the caller's concrete args
        // (e.g., `[f32]`) to get the actual monomorphized callee.
        //
        // Example:
        //   Caller: cuda_oxide_kernel_<hash>_scale::<f32> (args = [f32])
        //   Call in MIR: scale<T>(...)  (args = [T])
        //   After substitution: scale::<f32> (args = [f32])
        let args = self.tcx.instantiate_and_normalize_erasing_regions(
            caller.args,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(*args),
        );

        // Check if function is from a crate we should compile
        match self.should_collect_from_crate(*def_id) {
            CollectDecision::Collect => {
                // Continue processing below
            }
            CollectDecision::SkipIntentional => {
                // Silently skip (fmt, panicking, etc.)
                return;
            }
            CollectDecision::Forbidden {
                crate_name,
                fn_path,
            } => {
                // ERROR: Device code is calling a forbidden crate!
                // Build a formatted error box (68 char inner width)
                let border = "═".repeat(68);
                let empty_line = format!("║{:68}║", "");

                // Truncate fn_path if too long (max 48 chars to fit in box)
                let fn_display = if fn_path.len() > 48 {
                    format!("{}...", &fn_path[..45])
                } else {
                    fn_path.clone()
                };

                // Build the "From crate" line with proper padding
                let crate_line = format!("║ From crate: '{}'", crate_name);
                let crate_padded = format!("{:<69}║", crate_line);

                // Build the last line with proper padding
                let last_line = format!(
                    "║ The '{}' crate cannot run on GPU (requires OS/heap).",
                    crate_name
                );
                let last_padded = format!("{:<69}║", last_line);

                panic!(
                    "\n\n\
╔{border}╗
║{title:^68}║
╠{border}╣
║ Device code calls: {fn:<48}║
{crate_line}
{empty}
║ Only these crates are allowed in device code:                      ║
║   • Local crate (your kernel code)                                 ║
║   • cuda_device (GPU intrinsics)                                     ║
║   • core (no_std standard library)                                 ║
{empty}
{last_line}
╚{border}╝
\n",
                    border = border,
                    title = "CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE",
                    fn = fn_display,
                    crate_line = crate_padded,
                    empty = empty_line,
                    last_line = last_padded,
                );
            }
        }

        // Special handling for closure trait method calls (FnOnce::call_once, etc.)
        // When we see a call like `<Closure as FnOnce>::call_once`, we need to collect
        // the closure body directly, because:
        // 1. The trait method itself may not have MIR
        // 2. The mir-importer transforms these calls to direct closure body calls
        let fn_name = self.tcx.def_path_str(*def_id);
        if fn_name.contains("call_once")
            || fn_name.contains("call_mut")
            || fn_name.ends_with("::call")
        {
            // Check if any type arg is a closure
            for arg in args.iter() {
                if let Some(ty) = arg.as_type()
                    && let TyKind::Closure(closure_def_id, closure_substs) = ty.kind()
                {
                    // Found a closure - add its body to the collection
                    let typing_env = TypingEnv::fully_monomorphized();
                    if let Some(closure_instance) =
                        Instance::try_resolve(self.tcx, typing_env, *closure_def_id, closure_substs)
                            .ok()
                            .flatten()
                    {
                        let mangled = self.tcx.symbol_name(closure_instance).name.to_string();
                        if !self.seen.contains(&mangled) {
                            let closure_name = self.fqdn(*closure_def_id);
                            let export_name =
                                self.compute_export_name(&closure_name, closure_instance);

                            if self.verbose {
                                eprintln!(
                                    "[collector] Discovered closure body (via trait call): {} -> {}",
                                    closure_name, export_name
                                );
                            }

                            self.seen.insert(mangled);
                            self.worklist.push_back(CollectedFunction {
                                instance: closure_instance,
                                is_kernel: false,
                                export_name,
                            });
                        }
                    }
                    // Don't return - continue to try resolving the trait method too
                    // (even though it may fail, we still want to try)
                }
            }
        }

        // Try to resolve the instance with substitutions first, so we can
        // check if we've already seen THIS specific monomorphization
        let typing_env = TypingEnv::fully_monomorphized();
        let Some(resolved) = Instance::try_resolve(self.tcx, typing_env, *def_id, args)
            .ok()
            .flatten()
        else {
            return;
        };

        // Skip already-seen monomorphizations (use mangled name as unique key)
        let mangled = self.tcx.symbol_name(resolved).name.to_string();
        if self.seen.contains(&mangled) {
            return;
        }

        // Skip non-monomorphized instances (still have generic type parameters).
        // This happens when walking the generic definition's MIR - the call args
        // are still generic. We only want to collect concrete instantiations.
        if !is_fully_monomorphized(self.tcx, resolved) {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping non-monomorphized callee: {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Skip intrinsics and other special functions
        if !matches!(resolved.def, InstanceKind::Item(_)) {
            return;
        }

        // Check if this is a device extern declaration (FFI with external LTOIR).
        // These have no MIR body but should be emitted as LLVM `declare` statements.
        let raw_name = self.tcx.def_path_str(resolved.def_id());
        if is_device_extern_symbol(&raw_name) {
            self.add_device_extern(resolved.def_id(), &raw_name);
            return;
        }

        // Skip functions without MIR bodies (extern intrinsics like cuda_device::threadIdx_x).
        // These are handled specially by the terminator translator in mir-importer
        // which dispatches them to NVVM intrinsic operations.
        if !self.tcx.is_mir_available(resolved.def_id()) {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping extern/intrinsic (no MIR): {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Check if it has an unreachable body (intrinsic placeholder)
        if self.is_unreachable_body(resolved.def_id()) {
            if self.verbose {
                eprintln!(
                    "[collector] Skipping intrinsic (unreachable body): {}",
                    self.tcx.def_path_str(resolved.def_id())
                );
            }
            return;
        }

        // Use FQDN so the export name matches what the MIR translator
        // sees via `CrateDef::name()` on the call side.
        let name = self.fqdn(resolved.def_id());
        let export_name = self.compute_export_name(&name, resolved);

        if self.verbose {
            eprintln!("[collector] Discovered callee: {} -> {}", name, export_name);
        }

        self.seen.insert(mangled);
        self.worklist.push_back(CollectedFunction {
            instance: resolved,
            is_kernel: false,
            export_name,
        });
    }

    /// Determines if a function from a given crate should be collected.
    ///
    /// Returns a [`CollectDecision`] indicating:
    /// - `Collect`: Function should be collected for device compilation
    /// - `SkipIntentional`: Skip silently (core::fmt, core::panicking)
    /// - `Forbidden`: Error - function is from a forbidden crate (std, etc.)
    ///
    /// ## Kernel Entry Points (Cross-Crate Support)
    ///
    /// Kernels (detected via `is_kernel_symbol` from `reserved-oxide-symbols`)
    /// are allowed from ANY crate. This enables library crates to export
    /// generic kernels that get monomorphized when used in an application.
    ///
    /// ## Allowed Crates (for non-kernel callees)
    ///
    /// - Local crate: The user's kernel code
    /// - `cuda_device`: Our GPU intrinsics library
    /// - `core`: Standard library core (iterators, Option, etc.)
    /// - `alloc`: Heap allocation (if user has configured a GPU allocator)
    /// - Any crate reachable from a kernel (transitive closure)
    ///
    /// ## Intentionally Skipped
    ///
    /// - `core::fmt::*`: Format trait machinery uses function pointers
    /// - `core::panicking::*`: Panic handling (handled by panic=abort)
    ///
    /// ## Forbidden (Error)
    ///
    /// - `std`: OS, I/O, threads - can't run on GPU
    fn should_collect_from_crate(&self, def_id: DefId) -> CollectDecision {
        // Always collect from local crate
        if def_id.krate == LOCAL_CRATE {
            return CollectDecision::Collect;
        }

        let crate_name = self.tcx.crate_name(def_id.krate);
        let name_str = crate_name.as_str();

        // Check if this is a kernel entry point. Kernels can come from ANY
        // crate — this enables library crates to export generic kernels that
        // get monomorphized when used in an application.
        let fn_name = self.tcx.item_name(def_id);
        if is_kernel_symbol(fn_name.as_str()) {
            return CollectDecision::Collect;
        }

        // Forbidden crate: std (OS, I/O, threads) - absolutely can't run on GPU
        if name_str == "std" {
            let fn_path = self.tcx.def_path_str(def_id);
            return CollectDecision::Forbidden {
                crate_name: name_str.to_string(),
                fn_path,
            };
        }

        // Allowed external crates (no_std compatible)
        // - core: iterators, Option, Result, traits, etc.
        // - alloc: Vec, Box, String (if user has GPU allocator)
        // - cuda_device: our GPU intrinsics
        let allowed = matches!(name_str, "core" | "alloc" | "cuda_device" | "cuda-device");

        if !allowed {
            // For other external crates, we allow them if they're reachable from a kernel.
            // This enables cross-crate device functions. The key safety is:
            // 1. std is explicitly forbidden above
            // 2. Any crate that uses std won't compile to PTX anyway (missing symbols)
            // 3. User gets a clear link error if they try to use incompatible code
            //
            // This permissive approach enables:
            // - Library crates with device helper functions
            // - Math libraries (e.g., libm, num-traits)
            // - Custom device abstractions
            if self.verbose {
                let fn_path = self.tcx.def_path_str(def_id);
                eprintln!(
                    "[collector] Allowing function from external crate '{}': {}",
                    name_str, fn_path
                );
            }
            return CollectDecision::Collect;
        }

        // Filter out problematic modules (intentional skip, not an error)
        let path = self.tcx.def_path_str(def_id);

        // Skip formatting and panic machinery (uses FnPtr types we can't translate)
        if path.contains("::fmt::") || path.contains("::panicking::") {
            return CollectDecision::SkipIntentional;
        }

        // Skip precondition_check functions - these are UB check assertions that use
        // string types for panic messages. Since our queries return false for
        // RuntimeChecks(UbChecks), these functions are never actually called at runtime.
        // Example: core::num::<impl usize>::unchecked_sub::precondition_check
        if path.contains("precondition_check") {
            return CollectDecision::SkipIntentional;
        }

        // NOTE: We no longer skip arithmetic trait methods (Mul::mul, Add::add, etc.)
        // These become device functions with call overhead, but that's a separate
        // optimization issue (forced-inline on monomorphic small bodies).
        //
        // Legacy mangled names (from prebuilt sysroot) contain $ characters which are
        // invalid PTX identifiers. We sanitize these in compute_export_name().

        CollectDecision::Collect
    }

    /// Computes the export name for a function.
    ///
    /// `name` must be the FQDN (from [`fqdn()`]) so that non-generic export names
    /// match what `CrateDef::name()` returns on the call side. The lowering layer
    /// converts `::` to `__` on both sides.
    ///
    /// For generic/monomorphized functions (or names with invalid PTX chars),
    /// we fall back to the mangled symbol name since PTX identifiers must match
    /// `[a-zA-Z_][a-zA-Z0-9_]*]`. The MIR translator also uses mangled names
    /// for generic calls, so both sides match.
    ///
    /// When pliron's `Legaliser` is adopted, the `::` to `__` conversion will
    /// be handled by the legaliser instead of manual replacement.
    fn compute_export_name(&mut self, name: &str, instance: Instance<'tcx>) -> String {
        let has_invalid_chars = name.contains('<')
            || name.contains('>')
            || name.contains('\'')
            || name.contains(' ')
            || name.contains('{')
            || name.contains('}')
            || name.contains('#');

        // CRITICAL: If the instance has generic args, we MUST use mangled name.
        // The MIR translator uses mangled names for generic function calls
        // (see terminator/mod.rs::extract_func_info), so we must match that here.
        // Without this, the call site uses "_RINv...mapf..." but we export as "map".
        let has_generic_args = !instance.args.is_empty();

        // Try the simple name first
        let simple_name = name.to_string();

        if has_invalid_chars || has_generic_args || self.used_export_names.contains(&simple_name) {
            // Use mangled symbol name to avoid conflicts
            // This handles generics (e.g., ptr::add::<i32>) and name collisions
            let mangled = self.tcx.symbol_name(instance).name.to_string();

            // Sanitize for PTX: replace $ with _ (legacy mangling uses $LT$, $GT$, etc.)
            let sanitized = sanitize_ptx_name(&mangled);

            self.used_export_names.insert(sanitized.clone());
            sanitized
        } else {
            self.used_export_names.insert(simple_name.clone());
            simple_name
        }
    }

    /// Checks if a function body is just `unreachable!()` (intrinsic placeholder).
    ///
    /// cuda_device intrinsics have placeholder bodies that panic when called on host:
    ///
    /// ```rust,ignore
    /// pub fn threadIdx_x() -> u32 {
    ///     unreachable!("threadIdx_x called outside CUDA kernel context")
    /// }
    /// ```
    ///
    /// These are translated specially to PTX intrinsics by the MIR translator.
    /// We skip collecting them because their panic bodies would pull in `FnPtr`
    /// types we can't handle.
    fn is_unreachable_body(&self, def_id: DefId) -> bool {
        if !self.tcx.is_mir_available(def_id) {
            return false;
        }

        let mir = self.tcx.optimized_mir(def_id);

        // Quick check: intrinsic bodies are very small (1-2 blocks)
        if mir.basic_blocks.len() > 2 {
            return false;
        }

        // Check for panic calls
        for bb_data in mir.basic_blocks.iter() {
            let Some(ref terminator) = bb_data.terminator else {
                continue;
            };
            match &terminator.kind {
                TerminatorKind::Call { func, .. } => {
                    if let Some(callee_def_id) = self.get_call_def_id(func) {
                        let path = self.tcx.def_path_str(callee_def_id);
                        // Match panic functions from both core (no_std) and std:
                        // - core::panicking::* (no_std mode)
                        // - std::rt::panic_fmt (std mode - unreachable!() expands to this)
                        if path.contains("::panicking::") || path.contains("::rt::panic") {
                            return true;
                        }
                    }
                }
                TerminatorKind::Unreachable => {}
                _ => return false,
            }
        }

        false
    }

    /// Extracts the DefId from a call operand.
    fn get_call_def_id(&self, func: &rustc_middle::mir::Operand<'tcx>) -> Option<DefId> {
        use rustc_middle::mir::Operand;

        let Operand::Constant(const_op) = func else {
            return None;
        };

        let ty = const_op.const_.ty();
        if let TyKind::FnDef(def_id, _) = ty.kind() {
            Some(*def_id)
        } else {
            None
        }
    }
}

/// Dumps MIR info for collected device functions.
///
/// This is useful for debugging to see what was collected and verify
/// the MIR statistics (basic blocks, locals, args) look reasonable.
pub fn dump_device_mir_info<'tcx>(tcx: TyCtxt<'tcx>, functions: &[CollectedFunction<'tcx>]) {
    eprintln!("\n=== Device Functions MIR Info ===");
    for func in functions {
        let def_id = func.instance.def_id();
        eprintln!(
            "\n{} [{}]:",
            func.export_name,
            if func.is_kernel { "kernel" } else { "device" }
        );

        if tcx.is_mir_available(def_id) {
            let mir = tcx.instance_mir(func.instance.def);
            eprintln!("  - {} basic blocks", mir.basic_blocks.len());
            eprintln!("  - {} local variables", mir.local_decls.len());
            eprintln!("  - {} args", mir.arg_count);

            // Show return type
            let ret_ty = mir.local_decls[rustc_middle::mir::RETURN_PLACE].ty;
            eprintln!("  - returns: {:?}", ret_ty);
        } else {
            eprintln!("  - MIR not available");
        }
    }
    eprintln!("=================================\n");
}
