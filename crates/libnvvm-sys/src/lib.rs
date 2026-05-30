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
//! 2. The system loader (`libnvvm.so.4`, `libnvvm.so.3`, `libnvvm.so`).
//! 3. `<root>/nvvm/lib64/libnvvm.so` for `<root>` in `CUDA_HOME`,
//!    `CUDA_PATH`, `/usr/local/cuda`, `/opt/cuda`.
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
use std::path::PathBuf;
use std::ptr;
use thiserror::Error;

// ============================================================================
// FFI types
// ============================================================================

/// Opaque libNVVM program handle (`nvvmProgram`).
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NvvmProgram(*mut c_void);

/// libNVVM result codes (`nvvmResult`). Mirrors `nvvm.h`.
#[allow(dead_code)]
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum NvvmResult {
    Success = 0,
    OutOfMemory = 1,
    ProgramCreationFailure = 2,
    IrVersionMismatch = 3,
    InvalidInput = 4,
    InvalidProgram = 5,
    InvalidIr = 6,
    InvalidOption = 7,
    NoModuleInProgram = 8,
    CompilationFailure = 9,
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
        "libnvvm.so could not be located. Set LIBNVVM_PATH or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
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
    compile_program: unsafe extern "C" fn(NvvmProgram, c_int, *const *const c_char) -> NvvmResult,
    get_compiled_result_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResult,
    get_compiled_result: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult,
    get_program_log_size: unsafe extern "C" fn(NvvmProgram, *mut usize) -> NvvmResult,
    get_program_log: unsafe extern "C" fn(NvvmProgram, *mut c_char) -> NvvmResult,
    get_error_string: unsafe extern "C" fn(NvvmResult) -> *const c_char,
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> NvvmResult,
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

/// Resolve an optional symbol to a function pointer of inferred type `T`.
///
/// Use this for libNVVM queries added after the baseline API that cuda-oxide
/// supports. Missing optional symbols preserve compatibility with older CUDA
/// Toolkit installations.
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
                compile_program: resolve(&lib, "nvvmCompileProgram")?,
                get_compiled_result_size: resolve(&lib, "nvvmGetCompiledResultSize")?,
                get_compiled_result: resolve(&lib, "nvvmGetCompiledResult")?,
                get_program_log_size: resolve(&lib, "nvvmGetProgramLogSize")?,
                get_program_log: resolve(&lib, "nvvmGetProgramLog")?,
                get_error_string: resolve(&lib, "nvvmGetErrorString")?,
                version: resolve(&lib, "nvvmVersion")?,
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

    /// Query the LLVM IR major version guaranteed by libNVVM for `arch`.
    ///
    /// `arch` must use the same form accepted by `nvvmCompileProgram`, such
    /// as `"compute_75"` or `"compute_120"`. CUDA 13+ libNVVM exposes
    /// `nvvmLLVMVersion` so callers can distinguish the LLVM 7 typed-pointer
    /// dialect from the modern opaque-pointer dialect for a concrete target.
    ///
    /// Returns `Ok(None)` when the loaded libNVVM predates this query.
    ///
    /// # Panics
    ///
    /// Panics if `arch` contains an interior NUL byte.
    pub fn llvm_version(&self, arch: &str) -> Result<Option<i32>, NvvmError> {
        let Some(llvm_version) = self.llvm_version else {
            return Ok(None);
        };

        let carch = CString::new(arch).expect("arch has interior NUL");
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
        if r != NvvmResult::Success || size <= 1 {
            return None;
        }
        let mut buf = vec![0u8; size];
        let r =
            unsafe { (self.nvvm.get_program_log)(self.handle, buf.as_mut_ptr() as *mut c_char) };
        if r != NvvmResult::Success {
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
    if r == NvvmResult::Success {
        return Ok(());
    }
    Err(NvvmError::Call {
        operation: op,
        code: r as i32,
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

    for soname in ["libnvvm.so.4", "libnvvm.so.3", "libnvvm.so"] {
        tried.push(soname.to_string());
        if let Ok(lib) = unsafe { Library::new(soname) } {
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

    None
}

fn cuda_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Ok(r) = std::env::var(var) {
            roots.push(PathBuf::from(r));
        }
    }
    roots.push(PathBuf::from("/usr/local/cuda"));
    roots.push(PathBuf::from("/opt/cuda"));
    roots
}
