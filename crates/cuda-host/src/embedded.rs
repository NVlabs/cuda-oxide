/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle,
    artifact_bundles_from_binary_path, artifact_bundles_from_current_exe,
    embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use std::ffi::{CStr, OsStr, c_char, c_int, c_void};
use std::mem::MaybeUninit;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the binary that was read.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// PTX source could not be parsed or edited safely while merging bundles.
    #[error("embedded CUDA module '{name}' contains invalid PTX: {reason}")]
    InvalidPtx { name: String, reason: String },

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are linked to an in-memory cubin for their original target.
/// A payload built for a standard pre-Blackwell target, such as `sm_86`, may
/// instead be converted to PTX and JIT-compiled by the driver on Blackwell.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load a named embedded artifact bundle from the binary that contains `addr`.
///
/// [`load_embedded_module`] reads the bundle from the current executable,
/// which is right only while the `#[cuda_module]` was compiled into that
/// executable. A module compiled into a shared object (a `cdylib` plugin, a
/// Python extension module, an evcxr cell) carries its bundle in that object,
/// and the process that later `dlopen`s it does not. This function asks the
/// dynamic loader which binary maps `addr` and reads the bundle from there.
/// The generated `load()` passes the address of its artifact anchor, which
/// the linker places in the same binary as the bundle.
///
/// When the loader has no absolute path for that binary (glibc names the
/// main program by its `argv[0]`, which may be relative or a bare name), the
/// current executable is read instead, exactly as [`load_embedded_module`]
/// would. Payload handling is the same as well.
pub fn load_embedded_module_from_address(
    ctx: &Arc<CudaContext>,
    name: &str,
    addr: *const u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_containing(addr)?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Artifact bundles of the binary that maps `addr`, falling back to the
/// current executable when the loader does not name that binary by an
/// absolute path.
fn artifact_bundles_containing(
    addr: *const u8,
) -> Result<Vec<OwnedArtifactBundle>, EmbeddedModuleError> {
    match binary_path_containing(addr) {
        Some(path) if path.is_absolute() => Ok(artifact_bundles_from_binary_path(path)?),
        _ => Ok(artifact_bundles_from_current_exe()?),
    }
}

/// Path of the executable or shared object that maps `addr`, as the dynamic
/// loader recorded it: the `dlopen` argument for a shared object, `argv[0]`
/// for the main program under glibc. `None` when no loaded object maps `addr`.
fn binary_path_containing(addr: *const u8) -> Option<PathBuf> {
    let mut info = MaybeUninit::<DlInfo>::zeroed();
    // SAFETY: `dladdr` reads nothing through `addr` and writes every field of
    // `info` when it returns non-zero; the all-zero start is itself a valid
    // `Dl_info` (null pointers, null base).
    let found = unsafe { dladdr(addr.cast(), info.as_mut_ptr()) };
    if found == 0 {
        return None;
    }
    // SAFETY: zero-initialised above and fully written by `dladdr`.
    let info = unsafe { info.assume_init() };
    if info.dli_fname.is_null() {
        return None;
    }
    // SAFETY: `dli_fname` points at a NUL-terminated string owned by the
    // loader, which keeps it alive while the object stays mapped; it is
    // copied before this function returns.
    let name = unsafe { CStr::from_ptr(info.dli_fname) };
    Some(PathBuf::from(OsStr::from_bytes(name.to_bytes())))
}

/// `Dl_info` from `<dlfcn.h>`: the object and symbol that map an address.
///
/// Declared here, like `dladdr` below, instead of through the `libc` crate:
/// a new dependency of cuda-host would invalidate the lockfile of every
/// example workspace, and this is the one call cuda-host makes.
#[repr(C)]
#[allow(dead_code)] // layout only; `dli_fname` is the one field read
struct DlInfo {
    dli_fname: *const c_char,
    dli_fbase: *mut c_void,
    dli_sname: *const c_char,
    dli_saddr: *mut c_void,
}

#[cfg_attr(not(target_feature = "crt-static"), link(name = "dl"))]
unsafe extern "C" {
    /// Fills `info` for the loaded object that maps `addr`; zero when none does.
    fn dladdr(addr: *const c_void, info: *mut DlInfo) -> c_int;
}

/// Merge all PTX bundles from the current executable into a single CUDA module.
///
/// When a generic kernel is monomorphized in a consuming crate, its PTX ends
/// up in that crate's bundle rather than the defining crate's bundle. This
/// function gathers every PTX bundle in the binary, strips duplicate header
/// directives (`.version`, `.target`, `.address_size`) from all but the first
/// bundle, concatenates the bodies, and loads the result as one CUDA module.
/// All kernel symbols are therefore available regardless of which crate bundle
/// they were compiled into.
///
/// Bundles with non-PTX payloads (NVVM IR, LTOIR, cubin) are skipped; use
/// `load_embedded_module` for those.
pub fn load_all_ptx_bundles_merged(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_from_current_exe()?;

    let mut merged = String::new();
    let mut found_any = false;

    for bundle in &bundles {
        if let Some(ptx_bytes) = bundle.payload(ArtifactPayloadKind::Ptx) {
            let ptx_str = std::str::from_utf8(ptx_bytes)
                .map_err(|_| EmbeddedModuleError::UnsupportedPayload {
                    name: bundle.name.clone(),
                })?
                .trim_end_matches('\0');

            if !found_any {
                merged.push_str(ptx_str);
                merged.push('\n');
                found_any = true;
            } else {
                // Strip per-file header directives; only one set is valid in a
                // concatenated PTX module.
                let body = strip_ptx_module_headers(ptx_str).map_err(|reason| {
                    EmbeddedModuleError::InvalidPtx {
                        name: bundle.name.clone(),
                        reason,
                    }
                })?;
                merged.push_str(&body);
                if !body.ends_with('\n') {
                    merged.push('\n');
                }
            }
        }
    }

    if !found_any {
        return Err(EmbeddedModuleError::NoModules);
    }

    let module = ctx.load_module_from_image(merged.as_bytes())?;
    // Retain the merged module's `.entry` names (a few dozen bytes per
    // kernel). If a later `_TID_` generic-kernel lookup misses while a
    // same-base entry exists under a different hash, the launch paths can
    // then report a host/device type-identity naming divergence instead of an
    // opaque "named symbol not found". See `crate::entry_registry`.
    crate::entry_registry::register_merged_module_entries(&module, &merged);
    Ok(module)
}

fn strip_ptx_module_headers(ptx: &str) -> Result<String, String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = ptx_parse::EditScript::new();
    for directive in document
        .directives()
        .iter()
        .filter(|directive| matches!(directive.name(), ".version" | ".target" | ".address_size"))
    {
        edits
            .delete(directive.line_span())
            .map_err(|error| error.to_string())?;
    }
    edits.apply(ptx).map_err(|error| error.to_string())
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::build_cubin_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::build_ptx_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::link_ltoir_to_cubin_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::link_ltoir_to_ptx_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(
    bundle: &OwnedArtifactBundle,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_bundle_with_explicit(bundle, explicit.as_deref())
}

fn target_arch_for_bundle_with_explicit(
    bundle: &OwnedArtifactBundle,
    explicit_target: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    ltoir::resolve_source_target(concrete_bundle_target(&bundle.target)?, explicit_target)
}

fn concrete_bundle_target(
    target: &str,
) -> Result<Option<cuda_artifact_finalizer::CudaArch>, ltoir::LtoirError> {
    match target {
        // Compatibility for artifacts emitted before concrete NVVM targets
        // were recorded. New bundles must never use these sentinels.
        "libdevice" | "nvvm-ir" => Ok(None),
        target => Ok(Some(target.parse()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        assert_eq!(
            concrete_bundle_target("sm_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn strips_only_structural_ptx_module_headers() {
        let ptx = "\
// .target sm_1
.version 8.9
.target sm_120a, debug
.address_size 64
.visible .entry kernel() { ret; }
";
        assert_eq!(
            strip_ptx_module_headers(ptx).unwrap(),
            "// .target sm_1\n.visible .entry kernel() { ret; }\n"
        );
    }

    #[test]
    fn target_arch_uses_bundle_compute_target() {
        assert_eq!(
            concrete_bundle_target("compute_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        // Older bundles used these names instead of recording a concrete
        // architecture. New bundles always record a validated `sm_*` target.
        for legacy in ["libdevice", "nvvm-ir"] {
            assert_eq!(concrete_bundle_target(legacy).unwrap(), None);
        }
    }

    #[test]
    fn legacy_bundle_without_recorded_target_requires_explicit_target() {
        for sentinel in ["libdevice", "nvvm-ir"] {
            let bundle = bundle_with_target(sentinel);
            let error = target_arch_for_bundle_with_explicit(&bundle, None)
                .expect_err("the original build target is required");
            assert!(matches!(error, ltoir::LtoirError::TargetNotFound));

            let asserted =
                target_arch_for_bundle_with_explicit(&bundle, Some("compute_86")).unwrap();
            assert_eq!(asserted.sm(), "sm_86");
        }
    }

    #[test]
    fn recorded_bundle_target_overrides_explicit_environment_target() {
        let bundle = bundle_with_target("sm_90");
        let selected = target_arch_for_bundle_with_explicit(&bundle, Some("sm_120")).unwrap();
        assert_eq!(selected.sm(), "sm_90");
    }

    #[test]
    fn target_arch_rejects_malformed_bundle_target() {
        assert!(concrete_bundle_target("sm_90x").is_err());
    }

    fn canonical(path: &std::path::Path) -> PathBuf {
        std::fs::canonicalize(path).expect("path exists")
    }

    #[test]
    fn address_in_the_executable_resolves_to_the_current_exe() {
        // Same shape as the artifact anchor: a static linked into the binary.
        static ANCHOR: u8 = 0;
        let path = binary_path_containing(std::ptr::addr_of!(ANCHOR))
            .expect("the test executable is a loaded object");
        assert_eq!(
            canonical(&path),
            canonical(&std::env::current_exe().unwrap())
        );
    }

    #[test]
    fn address_in_a_shared_object_resolves_to_that_object() {
        #[cfg_attr(not(target_feature = "crt-static"), link(name = "dl"))]
        unsafe extern "C" {
            fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
        }
        // `RTLD_DEFAULT` from `<dlfcn.h>`: search every loaded object.
        let rtld_default: *mut c_void = std::ptr::null_mut();
        // `getpid` lives in the C library, a shared object that is not the
        // test executable; `dlsym` returns its address in there.
        let symbol = c"getpid";
        // SAFETY: `RTLD_DEFAULT` is a valid pseudo-handle and `symbol` is
        // NUL-terminated.
        let addr = unsafe { dlsym(rtld_default, symbol.as_ptr()) };
        assert!(!addr.is_null());
        let path =
            binary_path_containing(addr as *const u8).expect("the C library is a loaded object");
        assert!(path.is_absolute(), "{}", path.display());
        assert!(path.exists(), "{}", path.display());
        assert_ne!(
            canonical(&path),
            canonical(&std::env::current_exe().unwrap())
        );
    }

    #[test]
    fn bundles_of_an_executable_address_match_the_current_exe() {
        static ANCHOR: u8 = 0;
        let by_address = artifact_bundles_containing(std::ptr::addr_of!(ANCHOR))
            .map(|bundles| bundles.len())
            .map_err(|error| error.to_string());
        let by_current_exe = artifact_bundles_from_current_exe()
            .map(|bundles| bundles.len())
            .map_err(|error| error.to_string());
        assert_eq!(by_address, by_current_exe);
    }
}
