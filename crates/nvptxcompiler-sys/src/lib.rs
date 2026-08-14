/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Runtime (`dlopen`) bindings to NVIDIA's PTX Compiler API.
//!
//! The API assembles one PTX module into target-specific device code without
//! loading the CUDA Driver. [`LibNvPtxCompiler::load`] discovers the Toolkit
//! library at run time; [`Program`] owns one compiler handle.

use libloading::Library;
use std::ffi::{CString, c_char, c_int, c_void};
use std::fs::File;
#[cfg(target_os = "linux")]
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::SystemTime;
use thiserror::Error;

type ResultCode = i32;
type Handle = *mut c_void;
type Create = unsafe extern "C" fn(*mut Handle, usize, *const c_char) -> ResultCode;
type Destroy = unsafe extern "C" fn(*mut Handle) -> ResultCode;
type Compile = unsafe extern "C" fn(Handle, c_int, *const *const c_char) -> ResultCode;
type GetSize = unsafe extern "C" fn(Handle, *mut usize) -> ResultCode;
type GetProgram = unsafe extern "C" fn(Handle, *mut c_void) -> ResultCode;
type GetLog = unsafe extern "C" fn(Handle, *mut c_char) -> ResultCode;

/// Failures surfaced by the PTX Compiler API binding.
#[derive(Debug, Error)]
pub enum NvPtxCompilerError {
    /// No candidate shared library could be loaded completely.
    #[error("libnvptxcompiler could not be located; tried {tried}")]
    LibraryNotFound {
        /// Candidate paths and their loader errors.
        tried: String,
    },

    /// An option could not be represented by the C API.
    #[error("compiler option contains an interior NUL byte: {option:?}")]
    InvalidOption {
        /// Rejected option.
        option: String,
    },

    /// An API operation returned a non-success status.
    #[error("nvPTXCompiler error in {operation}: status {status}{}", .log.as_ref().map(|log| format!("\n--- nvPTXCompiler log ---\n{log}")).unwrap_or_default())]
    Call {
        /// API operation that failed.
        operation: &'static str,
        /// Raw result code, preserved for forward compatibility.
        status: ResultCode,
        /// Best-effort compiler error log.
        log: Option<String>,
    },
}

/// Loaded nvPTXCompiler library and immutable function table.
pub struct LibNvPtxCompiler {
    create: Create,
    destroy: Destroy,
    compile: Compile,
    get_program_size: GetSize,
    get_program: GetProgram,
    get_error_log_size: GetSize,
    get_error_log: GetLog,
    get_info_log_size: GetSize,
    get_info_log: GetLog,
    loaded_file: Option<File>,
    loaded_identity: Option<LibraryFileIdentity>,
    _library: Library,
}

// SAFETY: The library and immutable function table are safe to share. Each
// `Program` owns a distinct API handle and remains neither Send nor Sync.
unsafe impl Send for LibNvPtxCompiler {}
unsafe impl Sync for LibNvPtxCompiler {}

impl LibNvPtxCompiler {
    /// Discover and load nvPTXCompiler without retaining a fingerprintable file.
    pub fn load() -> Result<Self, NvPtxCompilerError> {
        Self::load_inner(false)
    }

    /// Load nvPTXCompiler and retain the exact opened descriptor on Linux.
    #[doc(hidden)]
    pub fn load_for_cache() -> Result<Self, NvPtxCompilerError> {
        Self::load_inner(true)
    }

    fn load_inner(retain_exact_file: bool) -> Result<Self, NvPtxCompilerError> {
        let mut tried = Vec::new();
        for candidate in library_candidates() {
            match unsafe { load_from_path(&candidate, retain_exact_file) } {
                Ok(library) => return Ok(library),
                Err(error) => tried.push(format!("{}: {error}", candidate.display())),
            }
        }
        Err(NvPtxCompilerError::LibraryNotFound {
            tried: tried.join(" | "),
        })
    }

    /// Exact retained descriptor, provided it has not changed since loading.
    #[doc(hidden)]
    pub fn loaded_file_if_unchanged(&self) -> Option<&File> {
        let identity = self.loaded_identity.as_ref()?;
        let file = self.loaded_file.as_ref()?;
        identity.matches_file(file).then_some(file)
    }
}

/// One nvPTXCompiler program handle.
pub struct Program<'a> {
    library: &'a LibNvPtxCompiler,
    handle: Handle,
}

impl<'a> Program<'a> {
    /// Create a program from logical, non-NUL-terminated PTX bytes.
    pub fn new(library: &'a LibNvPtxCompiler, ptx: &[u8]) -> Result<Self, NvPtxCompilerError> {
        let mut handle = ptr::null_mut();
        let status = unsafe { (library.create)(&mut handle, ptx.len(), ptx.as_ptr().cast()) };
        if status != 0 {
            return Err(NvPtxCompilerError::Call {
                operation: "nvPTXCompilerCreate",
                status,
                log: None,
            });
        }
        Ok(Self { library, handle })
    }

    /// Compile with options in exact supplied order.
    pub fn compile(&mut self, options: &[&str]) -> Result<(), NvPtxCompilerError> {
        let storage = options
            .iter()
            .map(|option| {
                CString::new(*option).map_err(|_| NvPtxCompilerError::InvalidOption {
                    option: (*option).to_owned(),
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let pointers = storage
            .iter()
            .map(|option| option.as_ptr())
            .collect::<Vec<_>>();
        let status = unsafe {
            (self.library.compile)(self.handle, pointers.len() as c_int, pointers.as_ptr())
        };
        self.check(status, "nvPTXCompilerCompile")
    }

    /// Retrieve the complete compiled program bytes.
    pub fn compiled_program(&self) -> Result<Vec<u8>, NvPtxCompilerError> {
        let mut size = 0usize;
        let status = unsafe { (self.library.get_program_size)(self.handle, &mut size) };
        self.check(status, "nvPTXCompilerGetCompiledProgramSize")?;
        let mut image = vec![0u8; size];
        let status = unsafe { (self.library.get_program)(self.handle, image.as_mut_ptr().cast()) };
        self.check(status, "nvPTXCompilerGetCompiledProgram")?;
        Ok(image)
    }

    /// Best-effort informational compiler log.
    pub fn info_log(&self) -> Option<String> {
        self.log(self.library.get_info_log_size, self.library.get_info_log)
    }

    /// Best-effort error compiler log.
    pub fn error_log(&self) -> Option<String> {
        self.log(self.library.get_error_log_size, self.library.get_error_log)
    }

    fn check(&self, status: ResultCode, operation: &'static str) -> Result<(), NvPtxCompilerError> {
        if status == 0 {
            Ok(())
        } else {
            Err(NvPtxCompilerError::Call {
                operation,
                status,
                log: self.error_log(),
            })
        }
    }

    fn log(&self, get_size: GetSize, get_log: GetLog) -> Option<String> {
        let mut size = 0usize;
        if unsafe { get_size(self.handle, &mut size) } != 0 || size <= 1 {
            return None;
        }
        let mut bytes = vec![0u8; size];
        if unsafe { get_log(self.handle, bytes.as_mut_ptr().cast()) } != 0 {
            return None;
        }
        if bytes.last() == Some(&0) {
            bytes.pop();
        }
        Some(String::from_utf8_lossy(&bytes).into_owned())
    }
}

impl Drop for Program<'_> {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { (self.library.destroy)(&mut self.handle) };
        }
    }
}

unsafe fn load_from_path(
    path: &Path,
    retain_exact_file: bool,
) -> Result<LibNvPtxCompiler, libloading::Error> {
    #[cfg(not(target_os = "linux"))]
    let _ = retain_exact_file;
    #[cfg(target_os = "linux")]
    let retained = if retain_exact_file {
        path.canonicalize()
            .ok()
            .and_then(|path| File::open(path).ok())
            .filter(|file| file.metadata().is_ok_and(|metadata| metadata.is_file()))
    } else {
        None
    };
    #[cfg(not(target_os = "linux"))]
    let retained: Option<File> = None;

    #[cfg(target_os = "linux")]
    let load_path = retained
        .as_ref()
        .map(|file| PathBuf::from(format!("/proc/self/fd/{}", file.as_raw_fd())))
        .unwrap_or_else(|| path.to_owned());
    #[cfg(not(target_os = "linux"))]
    let load_path = path.to_owned();

    let library = unsafe { Library::new(load_path) }?;
    let loaded_identity = retained
        .as_ref()
        .and_then(LibraryFileIdentity::capture_file)
        .filter(|identity| {
            retained
                .as_ref()
                .is_some_and(|file| identity.matches_file(file))
        });
    macro_rules! symbol {
        ($name:literal, $ty:ty) => {{
            let value = unsafe { library.get::<$ty>(concat!($name, "\0").as_bytes()) }?;
            *value
        }};
    }
    Ok(LibNvPtxCompiler {
        create: symbol!("nvPTXCompilerCreate", Create),
        destroy: symbol!("nvPTXCompilerDestroy", Destroy),
        compile: symbol!("nvPTXCompilerCompile", Compile),
        get_program_size: symbol!("nvPTXCompilerGetCompiledProgramSize", GetSize),
        get_program: symbol!("nvPTXCompilerGetCompiledProgram", GetProgram),
        get_error_log_size: symbol!("nvPTXCompilerGetErrorLogSize", GetSize),
        get_error_log: symbol!("nvPTXCompilerGetErrorLog", GetLog),
        get_info_log_size: symbol!("nvPTXCompilerGetInfoLogSize", GetSize),
        get_info_log: symbol!("nvPTXCompilerGetInfoLog", GetLog),
        loaded_file: retained,
        loaded_identity,
        _library: library,
    })
}

fn library_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    if let Some(path) = std::env::var_os("LIBNVPTXCOMPILER_PATH") {
        candidates.push(PathBuf::from(path));
    }
    for variable in ["CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_PATH"] {
        if let Some(root) = std::env::var_os(variable) {
            candidates.push(PathBuf::from(root).join("lib64/libnvptxcompiler.so"));
        }
    }
    for root in ["/usr/local/cuda", "/opt/cuda"] {
        candidates.push(PathBuf::from(root).join("lib64/libnvptxcompiler.so"));
    }
    candidates.extend(
        ["libnvptxcompiler.so.13", "libnvptxcompiler.so"]
            .into_iter()
            .map(PathBuf::from),
    );
    candidates
}

#[derive(Debug, Eq, PartialEq)]
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
        let metadata = file.metadata().ok()?;
        #[cfg(unix)]
        use std::os::unix::fs::MetadataExt;
        Some(Self {
            len: metadata.len(),
            modified: metadata.modified().ok()?,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn result_representation_accepts_future_error_codes() {
        let error = NvPtxCompilerError::Call {
            operation: "future operation",
            status: i32::MAX,
            log: None,
        };
        assert!(error.to_string().contains(&i32::MAX.to_string()));
    }

    #[test]
    fn retained_identity_detects_replaced_file() {
        let nonce = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "nvptxcompiler-sys-identity-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        let path = directory.join("tool.so");
        let replacement = directory.join("replacement.so");
        std::fs::write(&path, b"original").unwrap();
        std::fs::write(&replacement, b"replacement-with-another-length").unwrap();
        let opened = File::open(&path).unwrap();
        let identity = LibraryFileIdentity::capture_file(&opened).unwrap();
        assert!(identity.matches_file(&opened));
        std::fs::rename(&replacement, &path).unwrap();
        assert!(identity.matches_file(&opened));
        assert_ne!(
            LibraryFileIdentity::capture_file(&File::open(&path).unwrap()).unwrap(),
            identity
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
