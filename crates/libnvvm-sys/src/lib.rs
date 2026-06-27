/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA's libNVVM.
//!
//! libNVVM is the front-end of NVIDIA's PTX-targeting compiler. It accepts
//! NVVM IR (an LLVM-IR dialect) and produces either PTX or LTOIR.
//!
//! This crate is a thin, RAII Rust binding that loads `libnvvm.so` lazily
//! at runtime via `libloading`. It is not a `bindgen`-generated wrapper, so
//! it does not require the CUDA Toolkit to be present at build time, only
//! at run time.
//!
//! # Library discovery
//!
//! [`LibNvvm::load`] tries (in order):
//! 1. `LIBNVVM_PATH` env var, if set.
//! 2. `<root>/nvvm/lib64/libnvvm.so` for `<root>` in `CUDA_TOOLKIT_PATH`,
//!    `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`.
//! 3. The system loader (`libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so`).
//!
//! # Symbol naming
//!
//! libNVVM uses plain unversioned symbol names (`nvvmCreateProgram` etc.),
//! so a single `dlsym` lookup per function is sufficient across CUDA
//! versions.
//!
//! # Example
//!
//! ```no_run
//! use libnvvm_sys::{LibNvvm, Program};
//!
//! let nvvm = LibNvvm::load().expect("CUDA Toolkit (libnvvm) not found");
//! let mut program = Program::new(&nvvm).unwrap();
//! program.add_module(b"; NVVM IR here\n", "kernel").unwrap();
//! let ltoir = program.compile(&["-arch=compute_120", "-gen-lto"]).unwrap();
//! assert!(!ltoir.is_empty());
//! ```

use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_void};
use std::fmt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::str::FromStr;
use thiserror::Error;

// ============================================================================
// CUDA architecture
// ============================================================================

/// A validated CUDA compute capability, independent of its textual prefix.
///
/// libNVVM takes `compute_XX`, while cubin-producing nvJitLink calls take
/// `sm_XX`. Keeping one parsed value prevents those two consumers from
/// accidentally targeting different devices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CudaArch {
    capability: u32,
    suffix: Option<char>,
}

impl CudaArch {
    /// Numeric CUDA capability (`86`, `90`, `100`, `120`, ...).
    pub fn capability(&self) -> u32 {
        self.capability
    }

    /// Optional architecture-family suffix (`a` or `f`).
    ///
    /// Targets such as `sm_90a` enable architecture-specific instructions and
    /// cannot be forwarded to a different compute capability.
    pub fn suffix(&self) -> Option<char> {
        self.suffix
    }

    /// Whether libNVVM selects its legacy LLVM 7 input dialect.
    pub fn uses_legacy_llvm(&self) -> bool {
        self.capability < 100
    }

    /// Render the target for cubin-producing tools such as nvJitLink.
    pub fn sm(&self) -> String {
        self.render("sm_")
    }

    /// Render the target for libNVVM.
    pub fn compute(&self) -> String {
        self.render("compute_")
    }

    fn render(&self, prefix: &str) -> String {
        match self.suffix {
            Some(suffix) => format!("{prefix}{}{suffix}", self.capability),
            None => format!("{prefix}{}", self.capability),
        }
    }
}

impl FromStr for CudaArch {
    type Err = CudaArchParseError;

    fn from_str(target: &str) -> Result<Self, Self::Err> {
        let rest = target
            .strip_prefix("sm_")
            .or_else(|| target.strip_prefix("compute_"))
            .ok_or_else(|| CudaArchParseError::new(target, "expected `sm_XX` or `compute_XX`"))?;

        let digit_count = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digit_count < 2 {
            return Err(CudaArchParseError::new(
                target,
                "compute capability must contain at least two digits",
            ));
        }
        let (digits, suffix_text) = rest.split_at(digit_count);
        let suffix = match suffix_text {
            "" => None,
            "a" => Some('a'),
            "f" => Some('f'),
            _ => {
                return Err(CudaArchParseError::new(
                    target,
                    "the only supported architecture suffixes are `a` and `f`",
                ));
            }
        };
        let capability = digits.parse::<u32>().map_err(|_| {
            CudaArchParseError::new(target, "compute capability is not a valid integer")
        })?;

        Ok(Self { capability, suffix })
    }
}

impl fmt::Display for CudaArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.sm())
    }
}

/// A malformed CUDA architecture string.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
#[error("invalid CUDA target `{target}`: {reason}")]
pub struct CudaArchParseError {
    target: String,
    reason: &'static str,
}

impl CudaArchParseError {
    fn new(target: &str, reason: &'static str) -> Self {
        Self {
            target: target.to_string(),
            reason,
        }
    }
}

/// Versions accepted by the loaded libNVVM frontend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NvvmIrVersion {
    pub ir_major: i32,
    pub ir_minor: i32,
    pub debug_major: i32,
    pub debug_minor: i32,
}

// ============================================================================
// FFI types
// ============================================================================

/// Opaque libNVVM program handle (`nvvmProgram`).
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NvvmProgram(*mut c_void);

/// Integer representation of libNVVM's C `nvvmResult` enum.
///
/// This is an integer rather than a Rust enum so result codes added by newer
/// libNVVM versions remain valid values.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct NvvmResult(c_int);

impl NvvmResult {
    const SUCCESS: Self = Self(0);
    /// Present in CUDA 13.0 and newer headers.
    #[allow(dead_code)]
    const CANCELLED: Self = Self(10);
}

// ============================================================================
// Errors
// ============================================================================

/// All errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum NvvmError {
    /// `libnvvm.so` could not be located on this system. `tried` lists every
    /// path or SONAME that was probed, in order, joined by newlines.
    #[error(
        "libnvvm.so could not be located. Set LIBNVVM_PATH, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibraryNotFound {
        /// Newline-joined list of paths and SONAMEs that were probed.
        tried: String,
    },

    /// `libnvvm.so` was loaded, but `dlsym` failed to resolve a function this
    /// crate requires. Indicates an old or broken libNVVM that does not
    /// export the standard NVVM IR API.
    #[error("libnvvm.so was found but a required symbol is missing: {symbol}: {source}")]
    SymbolNotFound {
        /// Name of the missing libNVVM function (e.g. `nvvmCreateProgram`).
        symbol: &'static str,
        /// Underlying `libloading` error returned by `dlsym`.
        #[source]
        source: libloading::Error,
    },

    /// A libNVVM call returned a non-`Success` `nvvmResult`. `log` carries
    /// the libNVVM program log when it is available, or the
    /// `nvvmGetErrorString` text otherwise.
    #[error("libnvvm error in {operation}: {code:?}{}", .log.as_ref().map(|l| format!("\n--- libNVVM log ---\n{l}")).unwrap_or_default())]
    Call {
        /// Name of the libNVVM function that failed.
        operation: &'static str,
        /// Raw `nvvmResult` integer.
        code: i32,
        /// Best-effort error message: program log first, then
        /// `nvvmGetErrorString`. `None` only if both were unavailable.
        log: Option<String>,
    },
}

/// `libdevice.10.bc` could not be located on this system. `tried` lists
/// every path that was probed, in order, joined by newlines.
#[derive(Debug, Error)]
#[error(
    "Could not locate libdevice.10.bc. Set CUDA_OXIDE_LIBDEVICE, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
)]
pub struct LibdeviceNotFound {
    /// Newline-joined list of paths that were probed.
    pub tried: String,
}

// ============================================================================
// Library handle
// ============================================================================

/// Loaded libNVVM library plus resolved function pointers.
///
/// Hold one of these for the lifetime of any [`Program`] that borrows it.
/// `LibNvvm` owns the underlying `dlopen` handle; dropping it unloads the
/// library, which invalidates any function pointers obtained from it.
///
/// It is fine to call [`LibNvvm::load`] more than once if you want
/// independent handles; each call performs its own `dlopen` and resolves
/// its own symbols.
pub struct LibNvvm {
    _lib: Library,
    create_program: unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResult,
    destroy_program: unsafe extern "C" fn(*mut NvvmProgram) -> NvvmResult,
    add_module:
        unsafe extern "C" fn(NvvmProgram, *const c_char, usize, *const c_char) -> NvvmResult,
    verify_program: unsafe extern "C" fn(NvvmProgram, c_int, *const *const c_char) -> NvvmResult,
    compile_program: unsafe extern "C" fn(NvvmProgram, c_int, *const *const c_char) -> NvvmResult,
    get_compiled_result_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResult,
    get_compiled_result: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult,
    get_program_log_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResult,
    get_program_log: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult,
    get_error_string: unsafe extern "C" fn(NvvmResult) -> *const c_char,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvvmResult,
    ir_version: unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int, *mut c_int) -> NvvmResult,
    llvm_version: Option<unsafe extern "C" fn(*const c_char, *mut c_int) -> NvvmResult>,
}

// SAFETY: After `load()`, the struct contains only `extern "C"` function
// pointers and an owned `libloading::Library` handle. The function pointers
// are pure values and the library handle is `Send + Sync` (`libloading`
// guarantees this). libNVVM itself is internally synchronized for
// `nvvmProgram` operations on distinct programs, and we never share a single
// `Program` across threads (it does not implement `Send`).
unsafe impl Send for LibNvvm {}
unsafe impl Sync for LibNvvm {}

/// Resolve a symbol to a function pointer of inferred type `T`.
///
/// `T` is inferred from the field assignment context, so each `resolve(...)`
/// call at the [`LibNvvm::load`] site picks up the precise function-pointer
/// type of the field it is assigned to.
///
/// # Safety
///
/// The returned function pointer is valid only while the borrowed `lib`
/// remains loaded. Callers store the resolved pointer in [`LibNvvm`]
/// alongside the owning `Library`, so the pointer's lifetime matches the
/// `LibNvvm` instance.
unsafe fn resolve<T: Copy>(lib: &Library, name: &'static str) -> Result<T, NvvmError> {
    let sym: Symbol<T> =
        unsafe { lib.get(name.as_bytes()) }.map_err(|source| NvvmError::SymbolNotFound {
            symbol: name,
            source,
        })?;
    Ok(unsafe { *sym.into_raw() })
}

/// Resolve an optional symbol while remaining compatible with older toolkits.
unsafe fn resolve_optional<T: Copy>(lib: &Library, name: &'static str) -> Option<T> {
    let sym: Symbol<T> = unsafe { lib.get(name.as_bytes()) }.ok()?;
    Some(unsafe { *sym.into_raw() })
}

impl LibNvvm {
    /// Locate and load `libnvvm.so` at runtime, then resolve every libNVVM
    /// function this crate uses. Returns [`NvvmError::LibraryNotFound`] if
    /// none of the candidate paths could be opened, or
    /// [`NvvmError::SymbolNotFound`] if the loaded library is missing a
    /// required symbol.
    ///
    /// See the crate-level docs for the exact discovery order.
    pub fn load() -> Result<Self, NvvmError> {
        let mut tried = Vec::new();
        let lib = open_library(&mut tried).ok_or_else(|| NvvmError::LibraryNotFound {
            tried: tried.join("\n  "),
        })?;

        unsafe {
            Ok(LibNvvm {
                create_program: resolve(&lib, "nvvmCreateProgram")?,
                destroy_program: resolve(&lib, "nvvmDestroyProgram")?,
                add_module: resolve(&lib, "nvvmAddModuleToProgram")?,
                verify_program: resolve(&lib, "nvvmVerifyProgram")?,
                compile_program: resolve(&lib, "nvvmCompileProgram")?,
                get_compiled_result_size: resolve(&lib, "nvvmGetCompiledResultSize")?,
                get_compiled_result: resolve(&lib, "nvvmGetCompiledResult")?,
                get_program_log_size: resolve(&lib, "nvvmGetProgramLogSize")?,
                get_program_log: resolve(&lib, "nvvmGetProgramLog")?,
                get_error_string: resolve(&lib, "nvvmGetErrorString")?,
                version: resolve(&lib, "nvvmVersion")?,
                ir_version: resolve(&lib, "nvvmIRVersion")?,
                llvm_version: resolve_optional(&lib, "nvvmLLVMVersion"),
                _lib: lib,
            })
        }
    }

    /// Query libNVVM's version as `(major, minor)`. Wraps `nvvmVersion`,
    /// which returns the supported NVVM IR version (e.g. CUDA 13's libNVVM
    /// reports `(2, 0)`).
    ///
    /// Returns [`NvvmError::Call`] if the underlying call fails.
    pub fn version(&self) -> Result<(i32, i32), NvvmError> {
        let mut major = 0;
        let mut minor = 0;
        let r = unsafe { (self.version)(&mut major, &mut minor) };
        check(self, r, "nvvmVersion", None)?;
        Ok((major, minor))
    }

    /// Query the NVVM IR and debug-metadata versions accepted by libNVVM.
    pub fn ir_version(&self) -> Result<NvvmIrVersion, NvvmError> {
        let mut ir_major = 0;
        let mut ir_minor = 0;
        let mut debug_major = 0;
        let mut debug_minor = 0;
        let r = unsafe {
            (self.ir_version)(
                &mut ir_major,
                &mut ir_minor,
                &mut debug_major,
                &mut debug_minor,
            )
        };
        check(self, r, "nvvmIRVersion", None)?;
        Ok(NvvmIrVersion {
            ir_major,
            ir_minor,
            debug_major,
            debug_minor,
        })
    }

    /// Query the LLVM IR major version guaranteed by libNVVM for `arch`.
    ///
    /// CUDA 13+ libNVVM exposes
    /// `nvvmLLVMVersion` so callers can distinguish the LLVM 7 typed-pointer
    /// dialect from the modern opaque-pointer dialect for a concrete target.
    ///
    /// Returns `Ok(None)` when the loaded libNVVM predates this query.
    ///
    pub fn llvm_version(&self, arch: &CudaArch) -> Result<Option<i32>, NvvmError> {
        let Some(llvm_version) = self.llvm_version else {
            return Ok(None);
        };

        let carch = CString::new(arch.compute()).expect("rendered CUDA target contains NUL");
        let mut major = 0;
        let r = unsafe { llvm_version(carch.as_ptr(), &mut major) };
        check(self, r, "nvvmLLVMVersion", None)?;
        Ok(Some(major))
    }
}

// ============================================================================
// Program (RAII)
// ============================================================================

/// RAII wrapper around an `nvvmProgram` handle.
///
/// Typical usage:
///
/// 1. [`Program::new`] to create a fresh handle.
/// 2. One or more [`Program::add_module`] calls to feed in NVVM IR text or
///    LLVM bitcode (e.g. `libdevice.10.bc` plus the kernel module).
/// 3. [`Program::compile`] with libNVVM options (`-arch=...`, `-gen-lto`,
///    ...) to produce PTX or LTOIR bytes.
///
/// The handle is destroyed on drop. `Program` borrows the [`LibNvvm`] that
/// created it, so the library outlives every program handle.
pub struct Program<'a> {
    nvvm: &'a LibNvvm,
    handle: NvvmProgram,
}

impl<'a> Program<'a> {
    /// Create a fresh `nvvmProgram` handle. Wraps `nvvmCreateProgram`.
    pub fn new(nvvm: &'a LibNvvm) -> Result<Self, NvvmError> {
        let mut handle = NvvmProgram(ptr::null_mut());
        let r = unsafe { (nvvm.create_program)(&mut handle) };
        check(nvvm, r, "nvvmCreateProgram", None)?;
        Ok(Self { nvvm, handle })
    }

    /// Add an NVVM IR (text) or LLVM bitcode module to the program. Wraps
    /// `nvvmAddModuleToProgram`.
    ///
    /// `name` is recorded by libNVVM for use in diagnostic messages and
    /// program-log output. It does not need to correspond to a file on
    /// disk.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains an interior NUL byte.
    pub fn add_module(&mut self, ir: &[u8], name: &str) -> Result<(), NvvmError> {
        let cname = CString::new(name).expect("module name has interior NUL");
        let r = unsafe {
            (self.nvvm.add_module)(
                self.handle,
                ir.as_ptr() as *const c_char,
                ir.len(),
                cname.as_ptr(),
            )
        };
        let log = self.try_log();
        check(self.nvvm, r, "nvvmAddModuleToProgram", log)
    }

    /// Verify all modules for the supplied target and options, returning
    /// libNVVM's verifier log on failure.
    pub fn verify(&mut self, options: &[&str]) -> Result<(), NvvmError> {
        let coptions: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("option has interior NUL"))
            .collect();
        let optr: Vec<*const c_char> = coptions.iter().map(|s| s.as_ptr()).collect();

        let r =
            unsafe { (self.nvvm.verify_program)(self.handle, optr.len() as c_int, optr.as_ptr()) };
        let log = self.try_log();
        check(self.nvvm, r, "nvvmVerifyProgram", log)
    }

    /// Compile every previously-added module and return the produced PTX or
    /// LTOIR bytes. Wraps `nvvmCompileProgram` + `nvvmGetCompiledResult`.
    ///
    /// `options` are passed to libNVVM verbatim. Common choices:
    /// - `-arch=compute_XY` -- target compute capability (required).
    /// - `-gen-lto` -- emit LTOIR (instead of the default PTX).
    /// - `-opt=3` -- optimization level (`0`–`3`).
    ///
    /// On failure, returns [`NvvmError::Call`] with the libNVVM program log
    /// attached so the original NVVM diagnostic is preserved.
    ///
    /// # Panics
    ///
    /// Panics if any option string contains an interior NUL byte.
    pub fn compile(&mut self, options: &[&str]) -> Result<Vec<u8>, NvvmError> {
        let coptions: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("option has interior NUL"))
            .collect();
        let optr: Vec<*const c_char> = coptions.iter().map(|s| s.as_ptr()).collect();

        let r =
            unsafe { (self.nvvm.compile_program)(self.handle, optr.len() as c_int, optr.as_ptr()) };
        let log = self.try_log();
        check(self.nvvm, r, "nvvmCompileProgram", log)?;

        let mut size: usize = 0;
        let r = unsafe { (self.nvvm.get_compiled_result_size)(self.handle, &mut size) };
        check(self.nvvm, r, "nvvmGetCompiledResultSize", None)?;

        let mut buf = vec![0u8; size];
        let r = unsafe {
            (self.nvvm.get_compiled_result)(self.handle, buf.as_mut_ptr() as *mut c_char)
        };
        check(self.nvvm, r, "nvvmGetCompiledResult", None)?;

        Ok(buf)
    }

    /// Best-effort retrieval of the program log (warnings + errors).
    /// Returns `None` if the log is empty or cannot be fetched.
    fn try_log(&self) -> Option<String> {
        let mut size: usize = 0;
        let r = unsafe { (self.nvvm.get_program_log_size)(self.handle, &mut size) };
        if r != NvvmResult::SUCCESS || size <= 1 {
            return None;
        }
        let mut buf = vec![0u8; size];
        let r =
            unsafe { (self.nvvm.get_program_log)(self.handle, buf.as_mut_ptr() as *mut c_char) };
        if r != NvvmResult::SUCCESS {
            return None;
        }
        // Trim trailing NUL.
        if let Some(&0) = buf.last() {
            buf.pop();
        }
        Some(String::from_utf8_lossy(&buf).into_owned())
    }
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        unsafe {
            (self.nvvm.destroy_program)(&mut self.handle);
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn check(
    nvvm: &LibNvvm,
    r: NvvmResult,
    op: &'static str,
    log: Option<String>,
) -> Result<(), NvvmError> {
    if r == NvvmResult::SUCCESS {
        return Ok(());
    }
    Err(NvvmError::Call {
        operation: op,
        code: r.0,
        log: log.or_else(|| error_string(nvvm, r)),
    })
}

fn error_string(nvvm: &LibNvvm, r: NvvmResult) -> Option<String> {
    let p = unsafe { (nvvm.get_error_string)(r) };
    if p.is_null() {
        return None;
    }
    Some(
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned(),
    )
}

fn open_library(tried: &mut Vec<String>) -> Option<Library> {
    if let Ok(p) = std::env::var("LIBNVVM_PATH") {
        let path = PathBuf::from(&p);
        tried.push(path.display().to_string());
        if let Ok(lib) = unsafe { Library::new(&path) } {
            return Some(lib);
        }
    }

    for root in cuda_roots() {
        let path = root.join("nvvm/lib64/libnvvm.so");
        tried.push(path.display().to_string());
        if let Ok(lib) = unsafe { Library::new(&path) } {
            return Some(lib);
        }
    }

    for soname in ["libnvvm.so.4", "libnvvm.so.3", "libnvvm.so"] {
        tried.push(soname.to_string());
        if let Ok(lib) = unsafe { Library::new(soname) } {
            return Some(lib);
        }
    }

    None
}

fn cuda_roots() -> Vec<PathBuf> {
    cuda_roots_from_env(|var| std::env::var(var).ok())
}

fn cuda_roots_from_env(mut get_env: impl FnMut(&str) -> Option<String>) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(r) = get_env(var) {
            roots.push(PathBuf::from(r));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));
    roots
}

// ============================================================================
// libdevice discovery
// ============================================================================

/// Locate `libdevice.10.bc` from the CUDA Toolkit.
///
/// libdevice ships in the toolkit's `nvvm/` component alongside `libnvvm.so`
/// and is consumed together with libNVVM in the LTOIR pipeline, so its
/// discovery lives here next to the library discovery in [`LibNvvm::load`].
///
/// Search order:
/// 1. `CUDA_OXIDE_LIBDEVICE` env var (used as-is if it points to an
///    existing file).
/// 2. `<root>/nvvm/libdevice/libdevice.10.bc` for `<root>` in
///    `CUDA_TOOLKIT_PATH`, `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda`,
///    `/opt/cuda`.
///
/// Returns [`LibdeviceNotFound`] with the full list of probed paths if
/// nothing matches.
pub fn find_libdevice() -> Result<PathBuf, LibdeviceNotFound> {
    find_libdevice_with(|var| std::env::var(var).ok(), |path| path.exists())
}

fn find_libdevice_with(
    mut get_env: impl FnMut(&str) -> Option<String>,
    mut exists: impl FnMut(&Path) -> bool,
) -> Result<PathBuf, LibdeviceNotFound> {
    if let Some(p) = get_env("CUDA_OXIDE_LIBDEVICE") {
        let path = PathBuf::from(p);
        if exists(&path) {
            return Ok(path);
        }
    }
    let mut tried = Vec::new();
    for root in cuda_roots_from_env(&mut get_env) {
        let candidate = root.join("nvvm/libdevice/libdevice.10.bc");
        tried.push(candidate.display().to_string());
        if exists(&candidate) {
            return Ok(candidate);
        }
    }
    Err(LibdeviceNotFound {
        tried: tried.join("\n  "),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvvm_result_representation_accepts_cancelled_and_future_codes() {
        assert_eq!(NvvmResult::CANCELLED.0, 10);
        let future_code = NvvmResult(c_int::MAX);
        assert_ne!(future_code, NvvmResult::SUCCESS);
        assert_eq!(future_code.0, c_int::MAX);
    }

    #[test]
    fn cuda_arch_parses_and_renders_api_specific_spellings() {
        for (input, capability, suffix, sm, compute, legacy) in [
            ("sm_75", 75, None, "sm_75", "compute_75", true),
            ("compute_90a", 90, Some('a'), "sm_90a", "compute_90a", true),
            ("sm_100f", 100, Some('f'), "sm_100f", "compute_100f", false),
            ("compute_120", 120, None, "sm_120", "compute_120", false),
        ] {
            let arch: CudaArch = input.parse().unwrap();
            assert_eq!(arch.capability(), capability);
            assert_eq!(arch.suffix(), suffix);
            assert_eq!(arch.sm(), sm);
            assert_eq!(arch.compute(), compute);
            assert_eq!(arch.uses_legacy_llvm(), legacy);
        }
    }

    #[test]
    fn cuda_arch_rejects_ambiguous_or_malformed_targets() {
        for input in [
            "", "86", "sm_", "sm_9", "sm_90x", "sm_90aa", "SM_90", "gfx90a",
        ] {
            assert!(input.parse::<CudaArch>().is_err(), "{input}");
        }
    }

    #[test]
    #[ignore = "requires an installed CUDA Toolkit with libNVVM"]
    fn live_version_queries_and_legacy_verifier() {
        let nvvm = LibNvvm::load().unwrap();
        let version = nvvm.ir_version().unwrap();
        assert!(version.ir_major >= 1);
        assert!(version.debug_major >= 1);

        let arch: CudaArch = "compute_86".parse().unwrap();
        if let Some(llvm_major) = nvvm.llvm_version(&arch).unwrap() {
            assert_eq!(llvm_major, 7);
        }

        const LEGACY_MODULE: &[u8] = br#"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

define void @kernel() {
entry:
  ret void
}

!nvvm.annotations = !{!0}
!nvvmir.version = !{!1}
!0 = !{void ()* @kernel, !"kernel", i32 1}
!1 = !{i32 2, i32 0, i32 3, i32 1}
"#;
        let mut program = Program::new(&nvvm).unwrap();
        program
            .add_module(LEGACY_MODULE, "legacy-verifier")
            .unwrap();
        program.verify(&["-arch=compute_86"]).unwrap();
    }

    #[test]
    fn cuda_roots_prefers_project_toolkit_env_var() {
        let roots = cuda_roots_from_env(|var| match var {
            "CUDA_TOOLKIT_PATH" => Some("/cuda/toolkit".to_string()),
            "CUDA_HOME" => Some("/cuda/home".to_string()),
            "CUDA_PATH" => Some("/cuda/path".to_string()),
            _ => None,
        });

        assert_eq!(
            roots,
            vec![
                PathBuf::from("/cuda/toolkit"),
                PathBuf::from("/cuda/home"),
                PathBuf::from("/cuda/path"),
                PathBuf::from("/usr/local/cuda"),
                PathBuf::from("/opt/cuda"),
            ]
        );
    }

    #[test]
    fn find_libdevice_honors_explicit_override_file() {
        let found = find_libdevice_with(
            |var| (var == "CUDA_OXIDE_LIBDEVICE").then(|| "/elsewhere/libdevice.10.bc".to_string()),
            |path| path == Path::new("/elsewhere/libdevice.10.bc"),
        );

        assert_eq!(found.unwrap(), PathBuf::from("/elsewhere/libdevice.10.bc"));
    }

    #[test]
    fn find_libdevice_probes_roots_in_order() {
        // CUDA_HOME has the file, but CUDA_TOOLKIT_PATH is probed first and
        // also has it; the first match must win.
        let found = find_libdevice_with(
            |var| match var {
                "CUDA_TOOLKIT_PATH" => Some("/cuda/toolkit".to_string()),
                "CUDA_HOME" => Some("/cuda/home".to_string()),
                _ => None,
            },
            |path| {
                path == Path::new("/cuda/toolkit/nvvm/libdevice/libdevice.10.bc")
                    || path == Path::new("/cuda/home/nvvm/libdevice/libdevice.10.bc")
            },
        );

        assert_eq!(
            found.unwrap(),
            PathBuf::from("/cuda/toolkit/nvvm/libdevice/libdevice.10.bc")
        );
    }

    #[test]
    fn find_libdevice_failure_lists_every_probed_path() {
        let err = find_libdevice_with(
            |var| (var == "CUDA_HOME").then(|| "/cuda/home".to_string()),
            |_| false,
        )
        .unwrap_err();

        assert_eq!(
            err.tried,
            "/cuda/home/nvvm/libdevice/libdevice.10.bc\n  \
             /usr/local/cuda/nvvm/libdevice/libdevice.10.bc\n  \
             /opt/cuda/nvvm/libdevice/libdevice.10.bc"
        );
        let message = err.to_string();
        assert!(message.contains("CUDA_OXIDE_LIBDEVICE"));
        assert!(message.contains("CUDA_TOOLKIT_PATH"));
        assert!(message.contains("CUDA_HOME"));
    }
}
