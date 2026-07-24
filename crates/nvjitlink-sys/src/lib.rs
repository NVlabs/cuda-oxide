/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA's nvJitLink.
//!
//! nvJitLink links one or more LTOIR modules (and other input forms) into
//! a final cubin or PTX. It is part of the CUDA Toolkit and ships at
//! `<cuda>/lib64/libnvJitLink.so`.
//!
//! # Symbol naming
//!
//! `nvJitLink.h` `#define`s every public function to a versioned mangled
//! name, e.g. `nvJitLinkCreate -> __nvJitLinkCreate_13_0`, but the library
//! also exports the unversioned name with default ELF symbol versioning.
//! That means `dlsym(handle, "nvJitLinkCreate")` resolves to the right
//! function on every CUDA Toolkit version, so this binding does not need
//! to probe per-CUDA-version symbol suffixes.
//!
//! # Example
//!
//! ```no_run
//! use nvjitlink_sys::{LibNvJitLink, Linker, InputType};
//!
//! let nvj = LibNvJitLink::load().expect("CUDA Toolkit (nvJitLink) not found");
//! let mut linker = Linker::new(&nvj, &["-arch=sm_120", "-lto"]).unwrap();
//! let ltoir = std::fs::read("kernel.ltoir").unwrap();
//! linker.add(InputType::Ltoir, &ltoir, "kernel.ltoir").unwrap();
//! let cubin = linker.finish().unwrap();
//! ```

use libloading::{Library, Symbol};
use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::SystemTime;
use thiserror::Error;

// ============================================================================
// FFI types
// ============================================================================

/// Opaque nvJitLink handle (`nvJitLinkHandle`).
#[repr(transparent)]
#[derive(Copy, Clone)]
struct NvJitLinkHandle(*mut c_void);

/// Integer representation of nvJitLink's C `nvJitLinkResult` enum.
///
/// This is an integer rather than a Rust enum so result codes added by newer
/// nvJitLink versions remain valid values.
#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct NvJitLinkResult(c_int);

impl NvJitLinkResult {
    const SUCCESS: Self = Self(0);
}

/// nvJitLink input kinds (`nvJitLinkInputType`). Mirrors `nvJitLink.h`.
///
/// Pass to [`Linker::add`] to tell nvJitLink how to interpret a chunk of
/// input bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum InputType {
    /// Sentinel "no input" value. Not a valid argument to [`Linker::add`].
    None = 0,
    /// CUDA binary (cubin).
    Cubin = 1,
    /// PTX assembly.
    Ptx = 2,
    /// LTOIR — the output of libNVVM `compile(... "-gen-lto" ...)`.
    Ltoir = 3,
    /// CUDA fat binary.
    Fatbin = 4,
    /// Host object file.
    Object = 5,
    /// Host library archive.
    Library = 6,
    /// Index file (used with sliced fatbins).
    Index = 7,
    /// Auto-detect the kind from the bytes. Convenient but slower; prefer
    /// the specific variant when you know the input format.
    Any = 10,
}

// ============================================================================
// Errors
// ============================================================================

/// All errors surfaced by this crate.
#[derive(Debug, Error)]
pub enum NvJitLinkError {
    /// `libnvJitLink.so` could not be located on this system. `tried` lists
    /// every path or SONAME that was probed, in order, joined by newlines.
    #[error(
        "libnvJitLink.so could not be located. Set LIBNVJITLINK_PATH, CUDA_TOOLKIT_PATH, or CUDA_HOME, or install the CUDA Toolkit. Tried:\n  {tried}"
    )]
    LibraryNotFound {
        /// Newline-joined list of paths and SONAMEs that were probed.
        tried: String,
    },

    /// `libnvJitLink.so` was loaded, but `dlsym` failed to resolve a function
    /// this crate requires. Indicates an old or broken nvJitLink that does
    /// not export the standard linker API.
    #[error("libnvJitLink.so was found but a required symbol is missing: {symbol}: {source}")]
    SymbolNotFound {
        /// Name of the missing nvJitLink function (e.g. `nvJitLinkCreate`).
        symbol: &'static str,
        /// Underlying `libloading` error returned by `dlsym`.
        #[source]
        source: libloading::Error,
    },

    /// The loaded nvJitLink predates linked-PTX retrieval. Cubin output is
    /// still usable; only callers that explicitly request PTX receive this
    /// error.
    #[error(
        "the loaded libnvJitLink does not export nvJitLinkGetLinkedPtxSize/nvJitLinkGetLinkedPtx"
    )]
    PtxOutputUnavailable,

    /// An nvJitLink call returned a non-`Success` `nvJitLinkResult`. `log`
    /// carries the nvJitLink error log when one was produced by the call.
    #[error("nvJitLink error in {operation}: {code:?}{}", .log.as_ref().map(|l| format!("\n--- nvJitLink error log ---\n{l}")).unwrap_or_default())]
    Call {
        /// Name of the nvJitLink function that failed.
        operation: &'static str,
        /// Raw `nvJitLinkResult` integer.
        code: i32,
        /// nvJitLink error log, if available.
        log: Option<String>,
    },
}

// ============================================================================
// Library handle
// ============================================================================

/// Loaded nvJitLink library plus resolved function pointers.
///
/// Hold one of these for the lifetime of any [`Linker`] that borrows it.
/// `LibNvJitLink` owns the underlying `dlopen` handle; dropping it unloads
/// the library, which invalidates any function pointers obtained from it.
///
/// It is fine to call [`LibNvJitLink::load`] more than once if you want
/// independent handles; each call performs its own `dlopen` and resolves
/// its own symbols.
pub struct LibNvJitLink {
    _lib: Library,
    loaded_file: Option<File>,
    loaded_identity: Option<LibraryFileIdentity>,
    create:
        unsafe extern "C" fn(*mut NvJitLinkHandle, u32, *const *const c_char) -> NvJitLinkResult,
    destroy: unsafe extern "C" fn(*mut NvJitLinkHandle) -> NvJitLinkResult,
    add_data: unsafe extern "C" fn(
        NvJitLinkHandle,
        InputType,
        *const c_void,
        usize,
        *const c_char,
    ) -> NvJitLinkResult,
    complete: unsafe extern "C" fn(NvJitLinkHandle) -> NvJitLinkResult,
    get_linked_cubin_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_linked_cubin: unsafe extern "C" fn(NvJitLinkHandle, *mut c_void) -> NvJitLinkResult,
    get_linked_ptx_size:
        Option<unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult>,
    get_linked_ptx: Option<unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult>,
    get_error_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_error_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    get_info_log_size: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_info_log: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
    version: Option<unsafe extern "C" fn(*mut u32, *mut u32) -> NvJitLinkResult>,
}

// SAFETY: Same reasoning as `libnvvm-sys::LibNvvm`. The struct holds an
// owned `libloading::Library` (which is `Send + Sync`) and a set of
// `extern "C"` function pointers. We never share a single `Linker` across
// threads (it is not `Send`), so per-handle thread safety is not required
// from nvJitLink itself.
unsafe impl Send for LibNvJitLink {}
unsafe impl Sync for LibNvJitLink {}

/// Resolve a required symbol to a function pointer of inferred type `T`.
///
/// # Safety
///
/// The returned function pointer is valid only while the borrowed `lib`
/// remains loaded. Callers store the resolved pointer in [`LibNvJitLink`]
/// alongside the owning `Library`, so the pointer's lifetime matches the
/// `LibNvJitLink` instance.
unsafe fn resolve<T: Copy>(lib: &Library, name: &'static str) -> Result<T, NvJitLinkError> {
    let sym: Symbol<T> =
        unsafe { lib.get(name.as_bytes()) }.map_err(|source| NvJitLinkError::SymbolNotFound {
            symbol: name,
            source,
        })?;
    Ok(unsafe { *sym.into_raw() })
}

/// Resolve an optional symbol; returns `None` if missing.
///
/// Used for symbols that may not be present on older CUDA Toolkit versions
/// (e.g. `nvJitLinkVersion`, added in CTK 12.3).
///
/// # Safety
///
/// Same as [`resolve`].
unsafe fn resolve_optional<T: Copy>(lib: &Library, name: &'static str) -> Option<T> {
    let sym: Symbol<T> = unsafe { lib.get(name.as_bytes()) }.ok()?;
    Some(unsafe { *sym.into_raw() })
}

impl LibNvJitLink {
    /// Locate and load `libnvJitLink.so` at runtime, then resolve every
    /// nvJitLink function this crate uses.
    ///
    /// Returns [`NvJitLinkError::LibraryNotFound`] if none of the candidate
    /// paths could be opened, or [`NvJitLinkError::SymbolNotFound`] if the
    /// loaded library is missing a required symbol. See the crate-level
    /// docs for the exact discovery order.
    pub fn load() -> Result<Self, NvJitLinkError> {
        Self::load_inner(false)
    }

    /// Load nvJitLink while retaining an exact, fingerprintable descriptor
    /// when the platform supports it.
    ///
    /// This is intended for a process-wide pinned linker cache handle. On
    /// Linux it opens the concrete library before `dlopen` and retains that
    /// descriptor so callers can fingerprint the selected file. Callers must
    /// retain the returned `LibNvJitLink` for the process lifetime and restart
    /// to change toolkits. General callers should use [`LibNvJitLink::load`].
    #[doc(hidden)]
    pub fn load_for_cache() -> Result<Self, NvJitLinkError> {
        Self::load_inner(true)
    }

    fn load_inner(retain_exact_file: bool) -> Result<Self, NvJitLinkError> {
        let mut tried = Vec::new();
        let opened = open_library(&mut tried, retain_exact_file).ok_or_else(|| {
            NvJitLinkError::LibraryNotFound {
                tried: tried.join("\n  "),
            }
        })?;
        let OpenedLibrary {
            library: lib,
            loaded_file,
            loaded_identity,
        } = opened;

        unsafe {
            Ok(LibNvJitLink {
                create: resolve(&lib, "nvJitLinkCreate")?,
                destroy: resolve(&lib, "nvJitLinkDestroy")?,
                add_data: resolve(&lib, "nvJitLinkAddData")?,
                complete: resolve(&lib, "nvJitLinkComplete")?,
                get_linked_cubin_size: resolve(&lib, "nvJitLinkGetLinkedCubinSize")?,
                get_linked_cubin: resolve(&lib, "nvJitLinkGetLinkedCubin")?,
                // These symbols are optional so older toolkits continue to
                // support cubin output.
                get_linked_ptx_size: resolve_optional(&lib, "nvJitLinkGetLinkedPtxSize"),
                get_linked_ptx: resolve_optional(&lib, "nvJitLinkGetLinkedPtx"),
                get_error_log_size: resolve(&lib, "nvJitLinkGetErrorLogSize")?,
                get_error_log: resolve(&lib, "nvJitLinkGetErrorLog")?,
                get_info_log_size: resolve(&lib, "nvJitLinkGetInfoLogSize")?,
                get_info_log: resolve(&lib, "nvJitLinkGetInfoLog")?,
                version: resolve_optional(&lib, "nvJitLinkVersion"),
                loaded_file,
                loaded_identity,
                _lib: lib,
            })
        }
    }

    /// Return the exact file descriptor used to load nvJitLink, provided that
    /// its contents have not changed since `dlopen`.
    ///
    /// [`LibNvJitLink::load_for_cache`] opens concrete library paths before
    /// loading them and retains the descriptor. Callers may fingerprint it to
    /// bind cached linker output to the process-pinned tool. Ordinary
    /// [`LibNvJitLink::load`] calls return `None` here. Any `None` result means
    /// cache reuse must be skipped.
    #[doc(hidden)]
    pub fn loaded_file_if_unchanged(&self) -> Option<&File> {
        let identity = self.loaded_identity.as_ref()?;
        let file = self.loaded_file.as_ref()?;
        identity.matches_file(file).then_some(file)
    }

    /// Query nvJitLink's version as `(major, minor)`. Wraps
    /// `nvJitLinkVersion` (added in CTK 12.3).
    ///
    /// Returns `None` if the loaded library does not export
    /// `nvJitLinkVersion`, or if the call itself fails.
    pub fn version(&self) -> Option<(u32, u32)> {
        let f = self.version?;
        let mut major = 0;
        let mut minor = 0;
        let r = unsafe { f(&mut major, &mut minor) };
        if r == NvJitLinkResult::SUCCESS {
            Some((major, minor))
        } else {
            None
        }
    }
}

// ============================================================================
// Linker (RAII)
// ============================================================================

/// RAII wrapper around an `nvJitLinkHandle`.
///
/// Typical usage:
///
/// 1. [`Linker::new`] with the link options (`-arch=sm_XX`, `-lto`, ...).
/// 2. One or more [`Linker::add`] calls feeding LTOIR / PTX / cubin chunks.
/// 3. [`Linker::finish`] to drive the link and return the cubin bytes.
///
/// The handle is destroyed on drop. `Linker` borrows the [`LibNvJitLink`]
/// that created it, so the library outlives every linker handle.
pub struct Linker<'a> {
    nvj: &'a LibNvJitLink,
    handle: NvJitLinkHandle,
}

impl<'a> Linker<'a> {
    /// Create a fresh linker. Wraps `nvJitLinkCreate`.
    ///
    /// `options` are passed to nvJitLink verbatim. Common choices:
    /// - `-arch=sm_XY` -- target SM (required).
    /// - `-lto` -- enable link-time optimization (required to consume
    ///   LTOIR inputs).
    /// - `-time` / `-verbose` -- emit timing or info messages into the
    ///   nvJitLink info log.
    ///
    /// # Panics
    ///
    /// Panics if any option string contains an interior NUL byte.
    pub fn new(nvj: &'a LibNvJitLink, options: &[&str]) -> Result<Self, NvJitLinkError> {
        let coptions: Vec<CString> = options
            .iter()
            .map(|s| CString::new(*s).expect("option has interior NUL"))
            .collect();
        let optr: Vec<*const c_char> = coptions.iter().map(|s| s.as_ptr()).collect();

        let mut handle = NvJitLinkHandle(ptr::null_mut());
        let r = unsafe { (nvj.create)(&mut handle, optr.len() as u32, optr.as_ptr()) };
        check(
            nvj,
            &Linker {
                nvj,
                handle: NvJitLinkHandle(ptr::null_mut()),
            },
            r,
            "nvJitLinkCreate",
        )?;
        Ok(Self { nvj, handle })
    }

    /// Add a single input chunk (in `kind` format) to the link. Wraps
    /// `nvJitLinkAddData`.
    ///
    /// `name` is recorded by nvJitLink for use in diagnostic messages and
    /// info-log output. It does not need to correspond to a file on disk.
    ///
    /// # Panics
    ///
    /// Panics if `name` contains an interior NUL byte.
    pub fn add(&mut self, kind: InputType, data: &[u8], name: &str) -> Result<(), NvJitLinkError> {
        let cname = CString::new(name).expect("input name has interior NUL");
        let r = unsafe {
            (self.nvj.add_data)(
                self.handle,
                kind,
                data.as_ptr() as *const c_void,
                data.len(),
                cname.as_ptr(),
            )
        };
        check(self.nvj, self, r, "nvJitLinkAddData")
    }

    /// Drive the link and return the resulting cubin bytes. Wraps
    /// `nvJitLinkComplete` + `nvJitLinkGetLinkedCubin`.
    ///
    /// Consumes the [`Linker`]; on success the underlying handle is freed
    /// after the cubin has been copied out. On failure, the cubin is empty
    /// and the [`NvJitLinkError::Call`] carries the nvJitLink error log.
    ///
    /// If `CUDA_OXIDE_VERBOSE` is set in the environment, the nvJitLink
    /// info log (timings, sm_XY chosen, etc.) is forwarded to `stderr`.
    pub fn finish(self) -> Result<Vec<u8>, NvJitLinkError> {
        let r = unsafe { (self.nvj.complete)(self.handle) };
        check(self.nvj, &self, r, "nvJitLinkComplete")?;

        let mut size: usize = 0;
        let r = unsafe { (self.nvj.get_linked_cubin_size)(self.handle, &mut size) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedCubinSize")?;

        let mut buf = vec![0u8; size];
        let r =
            unsafe { (self.nvj.get_linked_cubin)(self.handle, buf.as_mut_ptr() as *mut c_void) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedCubin")?;

        // Forward the info log if anyone is listening (helpful with `-verbose`).
        if let Some(info) = self.try_info_log()
            && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some()
        {
            eprintln!("--- nvJitLink info log ---\n{info}");
        }

        Ok(buf)
    }

    /// Drive the link and return linked PTX text.
    ///
    /// Construct the linker with both `-lto` and `-ptx`. Unlike
    /// [`Self::finish`], this retrieves `nvJitLinkGetLinkedPtx*` output. The
    /// returned buffer may include nvJitLink's trailing NUL byte, which is
    /// accepted by the CUDA driver and useful for direct `cuModuleLoadData`.
    ///
    /// The PTX functions are optional so older nvJitLink versions can still
    /// produce cubins.
    pub fn finish_ptx(self) -> Result<Vec<u8>, NvJitLinkError> {
        let get_size = self
            .nvj
            .get_linked_ptx_size
            .ok_or(NvJitLinkError::PtxOutputUnavailable)?;
        let get = self
            .nvj
            .get_linked_ptx
            .ok_or(NvJitLinkError::PtxOutputUnavailable)?;

        let r = unsafe { (self.nvj.complete)(self.handle) };
        check(self.nvj, &self, r, "nvJitLinkComplete")?;

        let mut size = 0;
        let r = unsafe { get_size(self.handle, &mut size) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedPtxSize")?;

        let mut buf = vec![0u8; size];
        let r = unsafe { get(self.handle, buf.as_mut_ptr() as *mut c_char) };
        check(self.nvj, &self, r, "nvJitLinkGetLinkedPtx")?;

        if let Some(info) = self.try_info_log()
            && std::env::var_os("CUDA_OXIDE_VERBOSE").is_some()
        {
            eprintln!("--- nvJitLink info log ---\n{info}");
        }

        Ok(buf)
    }

    /// Best-effort retrieval of the error log.
    fn try_error_log(&self) -> Option<String> {
        try_log(
            self.nvj,
            self.handle,
            self.nvj.get_error_log_size,
            self.nvj.get_error_log,
        )
    }

    /// Best-effort retrieval of the info log.
    fn try_info_log(&self) -> Option<String> {
        try_log(
            self.nvj,
            self.handle,
            self.nvj.get_info_log_size,
            self.nvj.get_info_log,
        )
    }
}

impl Drop for Linker<'_> {
    fn drop(&mut self) {
        if !self.handle.0.is_null() {
            unsafe {
                (self.nvj.destroy)(&mut self.handle);
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

fn check(
    _nvj: &LibNvJitLink,
    linker: &Linker<'_>,
    r: NvJitLinkResult,
    op: &'static str,
) -> Result<(), NvJitLinkError> {
    if r == NvJitLinkResult::SUCCESS {
        return Ok(());
    }
    Err(NvJitLinkError::Call {
        operation: op,
        code: r.0,
        log: linker.try_error_log(),
    })
}

fn try_log(
    _nvj: &LibNvJitLink,
    handle: NvJitLinkHandle,
    size_fn: unsafe extern "C" fn(NvJitLinkHandle, *mut usize) -> NvJitLinkResult,
    get_fn: unsafe extern "C" fn(NvJitLinkHandle, *mut c_char) -> NvJitLinkResult,
) -> Option<String> {
    if handle.0.is_null() {
        return None;
    }
    let mut size: usize = 0;
    let r = unsafe { size_fn(handle, &mut size) };
    if r != NvJitLinkResult::SUCCESS || size <= 1 {
        return None;
    }
    let mut buf = vec![0u8; size];
    let r = unsafe { get_fn(handle, buf.as_mut_ptr() as *mut c_char) };
    if r != NvJitLinkResult::SUCCESS {
        return None;
    }
    if let Some(&0) = buf.last() {
        buf.pop();
    }
    Some(String::from_utf8_lossy(&buf).into_owned())
}

#[derive(Debug, PartialEq, Eq)]
struct LibraryFileIdentity {
    len: u64,
    modified: SystemTime,
    #[cfg(unix)]
    device: u64,
    #[cfg(unix)]
    inode: u64,
    #[cfg(unix)]
    change_time: (i64, i64),
}

impl LibraryFileIdentity {
    fn capture_file(file: &File) -> Option<Self> {
        Self::from_metadata(&file.metadata().ok()?)
    }

    fn from_metadata(metadata: &std::fs::Metadata) -> Option<Self> {
        let modified = metadata.modified().ok()?;

        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;

        Some(Self {
            len: metadata.len(),
            modified,
            #[cfg(unix)]
            device: metadata.dev(),
            #[cfg(unix)]
            inode: metadata.ino(),
            #[cfg(unix)]
            change_time: (metadata.ctime(), metadata.ctime_nsec()),
        })
    }

    fn matches_file(&self, file: &File) -> bool {
        Self::capture_file(file).as_ref() == Some(self)
    }

    #[cfg(test)]
    fn matches_path(&self, path: &Path) -> bool {
        path.metadata()
            .ok()
            .as_ref()
            .and_then(Self::from_metadata)
            .as_ref()
            == Some(self)
    }
}

struct OpenedLibrary {
    library: Library,
    loaded_file: Option<File>,
    loaded_identity: Option<LibraryFileIdentity>,
}

fn open_library(tried: &mut Vec<String>, retain_exact_file: bool) -> Option<OpenedLibrary> {
    if let Ok(p) = std::env::var("LIBNVJITLINK_PATH") {
        let path = PathBuf::from(&p);
        tried.push(path.display().to_string());
        if let Some(opened) = open_library_path(&path, retain_exact_file) {
            return Some(opened);
        }
    }

    for root in cuda_roots() {
        let path = root.join("lib64/libnvJitLink.so");
        tried.push(path.display().to_string());
        if let Some(opened) = open_library_path(&path, retain_exact_file) {
            return Some(opened);
        }
    }

    for soname in [
        "libnvJitLink.so.13",
        "libnvJitLink.so.12",
        "libnvJitLink.so",
    ] {
        tried.push(soname.to_string());
        if let Ok(lib) = unsafe { Library::new(soname) } {
            return Some(OpenedLibrary {
                library: lib,
                loaded_file: None,
                loaded_identity: None,
            });
        }
    }

    None
}

fn open_library_path(path: &Path, retain_exact_file: bool) -> Option<OpenedLibrary> {
    #[cfg(not(target_os = "linux"))]
    let _ = retain_exact_file;
    #[cfg(target_os = "linux")]
    let canonical_path = path.canonicalize().ok();

    #[cfg(target_os = "linux")]
    if retain_exact_file
        && let Some(canonical_path) = canonical_path.as_deref()
        && let Ok(file) = File::open(canonical_path)
        && file.metadata().is_ok_and(|metadata| metadata.is_file())
    {
        let identity = LibraryFileIdentity::capture_file(&file);
        // Load through the retained descriptor, not the pathname. A pathname
        // can already be present in glibc's dlopen cache for an older inode;
        // `/proc/self/fd/N` names the exact inode that we fingerprint below.
        let descriptor_path = PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd()));
        if let Ok(lib) = unsafe { Library::new(&descriptor_path) } {
            let identity = identity.filter(|identity| identity.matches_file(&file));
            return Some(OpenedLibrary {
                library: lib,
                loaded_file: Some(file),
                loaded_identity: identity,
            });
        }
    }

    let lib = unsafe { Library::new(path) }.ok()?;
    Some(OpenedLibrary {
        library: lib,
        // Loading by pathname cannot prove which mapping the dynamic loader
        // returned when another handle already exists for that pathname.
        loaded_file: None,
        loaded_identity: None,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_os = "linux")]
    fn compile_probe_library(source: &Path, output: &Path, value: i32) {
        std::fs::write(
            source,
            format!("int cuda_oxide_probe(void) {{ return {value}; }}\n"),
        )
        .unwrap();
        let status = std::process::Command::new("cc")
            .args(["-shared", "-fPIC", "-Wl,-soname,libprobe.so"])
            .arg(source)
            .arg("-o")
            .arg(output)
            .status()
            .expect("run C compiler for the dlopen identity regression test");
        assert!(status.success(), "C compiler failed with {status}");
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cache_loader_uses_replacement_inode_even_when_path_is_already_loaded() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nvjitlink-sys-dlopen-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let library_path = directory.join("libprobe.so");
        let replacement_path = directory.join("replacement.so");
        let first_source = directory.join("first.c");
        let second_source = directory.join("second.c");
        compile_probe_library(&first_source, &library_path, 1);

        let old_library = unsafe { Library::new(&library_path) }.unwrap();
        let old_probe: Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { old_library.get(b"cuda_oxide_probe") }.unwrap();
        assert_eq!(unsafe { old_probe() }, 1);

        compile_probe_library(&second_source, &replacement_path, 2);
        let replacement_bytes = std::fs::read(&replacement_path).unwrap();
        std::fs::rename(&replacement_path, &library_path).unwrap();

        let opened = open_library_path(&library_path, true).expect("load retained replacement");
        let replacement_probe: Symbol<unsafe extern "C" fn() -> c_int> =
            unsafe { opened.library.get(b"cuda_oxide_probe") }.unwrap();
        assert_eq!(unsafe { replacement_probe() }, 2);
        assert_eq!(unsafe { old_probe() }, 1);
        let retained = opened
            .loaded_file
            .as_ref()
            .expect("exact cache load retains its descriptor");
        let mut retained_bytes = Vec::new();
        std::io::Read::read_to_end(&mut retained.try_clone().unwrap(), &mut retained_bytes)
            .unwrap();
        assert_eq!(retained_bytes, replacement_bytes);
        assert!(opened.loaded_identity.is_some());

        drop(opened);
        drop(old_library);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn open_descriptor_remains_bound_to_replaced_inode() {
        let nonce = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nvjitlink-sys-identity-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let library_path = directory.join("libnvJitLink.so");
        let replacement_path = directory.join("replacement.so");
        std::fs::write(&library_path, b"original-library").unwrap();
        std::fs::write(
            &replacement_path,
            b"replacement-library-with-different-length",
        )
        .unwrap();

        let canonical_path = library_path.canonicalize().unwrap();
        let opened = File::open(&canonical_path).unwrap();
        let opened_identity = LibraryFileIdentity::capture_file(&opened).unwrap();
        assert!(opened_identity.matches_file(&opened));
        assert!(opened_identity.matches_path(&canonical_path));

        std::fs::remove_file(&library_path).unwrap();
        std::fs::rename(&replacement_path, &library_path).unwrap();
        assert!(!opened_identity.matches_path(&canonical_path));

        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;

            let opened_metadata = opened.metadata().unwrap();
            assert_eq!(opened_identity.device, opened_metadata.dev());
            assert_eq!(opened_identity.inode, opened_metadata.ino());
            let replacement_file = File::open(&canonical_path).unwrap();
            let replacement = LibraryFileIdentity::capture_file(&replacement_file).unwrap();
            assert_ne!(
                (opened_identity.device, opened_identity.inode),
                (replacement.device, replacement.inode)
            );
        }

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn result_representation_accepts_future_error_codes() {
        let future_code = NvJitLinkResult(c_int::MAX);
        assert_ne!(future_code, NvJitLinkResult::SUCCESS);
        assert_eq!(future_code.0, c_int::MAX);
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
    #[ignore = "requires an installed CUDA Toolkit with nvJitLink"]
    fn installed_toolkit_exposes_linked_ptx_output() {
        let library = LibNvJitLink::load().expect("load nvJitLink");
        assert!(library.get_linked_ptx_size.is_some());
        assert!(library.get_linked_ptx.is_some());
    }
}
