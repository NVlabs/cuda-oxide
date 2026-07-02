// SPDX-License-Identifier: Apache-2.0
//
// C ABI wrapper for the cuda-oxide library-mode in-process compiler.
//
// This crate packages the same in-process rustc_driver pipeline as
// `cuda-oxide-compiler` as a **cdylib** that exposes a small `extern "C"`
// surface, allowing a normal Rust app (no `rustc_private`, normal std) to
// `dlopen` this cdylib at runtime via `libloading` and drive cuda-oxide's
// in-process compiler: one library load, then many compiles, no subprocess.
//
// A cdylib is a COMPLETE, C-ABI artifact (like the bin, NOT an rlib library).
// It links `librustc_driver-<hash>.so` directly and bundles the plain rlibs
// (mir-importer/llvm-export/...) statically. We deliberately do NOT add an rlib
// crate-type: an rlib `rustc_private` library target poisons dependency-format
// resolution ("std only shows up once"); the cdylib resolves like the bin.
//
// NO DUPLICATION: the in-process compile logic is the SINGLE shared
// `compile_core` module, source-included via a relative `#[path]` to the bin
// crate's copy. The backend's two device modules are likewise source-included
// unchanged via `#[path]`. ZERO edits to `rustc_codegen_cuda`.

#![feature(rustc_private)]

extern crate rustc_abi;
extern crate rustc_driver;
extern crate rustc_hir;
extern crate rustc_interface;
extern crate rustc_middle;
extern crate rustc_public;
extern crate rustc_public_bridge;
extern crate rustc_session;
extern crate rustc_span;

#[path = "../../rustc-codegen-cuda/src/collector.rs"]
pub mod collector;
#[path = "../../rustc-codegen-cuda/src/device_codegen.rs"]
pub mod device_codegen;

// The SINGLE shared compile core -- the SAME file the bin includes, so the two
// targets cannot drift. (Relative `#[path]` into the sibling crate, mirroring
// the `#[path]` reuse of the backend's device modules above.)
#[path = "../../cuda-oxide-compiler/src/compile_core.rs"]
mod compile_core;

use std::ffi::{CStr, c_char, c_int};
use std::path::PathBuf;

use compile_core::{CompileRequest, compile_to_ptx, unique_out_dir};

// --------------------------------------------------------------------------
// C ABI -- the runtime entry points a normal app dlopens.
//
// NOTE: the C ABI is documented SINGLE-THREADED. One `cuda_oxide_compile` runs
// at a time per process; `compile_to_ptx` mutates process-global env
// (`CARGO_PKG_*`/`CUDA_OXIDE_TARGET`) the in-process rustc reads, which is sound
// only under that single-threaded contract.
// --------------------------------------------------------------------------

/// Read an optional NUL-terminated C string. `NULL` -> `None`; non-UTF-8 -> an
/// error the caller surfaces as a nonzero return.
unsafe fn opt_cstr<'a>(p: *const c_char) -> Result<Option<&'a str>, ()> {
    if p.is_null() {
        return Ok(None);
    }
    match unsafe { CStr::from_ptr(p) }.to_str() {
        Ok(s) => Ok(Some(s)),
        Err(_) => Err(()),
    }
}

/// Compile the kernel crate whose entry source file is `src_path` to PTX, fully
/// in-process.
///
/// Parameters (all NUL-terminated UTF-8 C strings):
///   * `src_path`      -- path to the kernel crate root `.rs` (required).
///   * `crate_name`    -- kernel crate name, feeds `CARGO_PKG_NAME` (required).
///   * `crate_version` -- kernel crate version, feeds `CARGO_PKG_VERSION`
///     (required).
///   * `arch`          -- device target e.g. "sm_80"; pass `NULL` to defer to
///     the pipeline's auto-detection.
///
/// On success: returns 0, writes a heap-allocated PTX buffer pointer to
/// `*out_ptr` and its length to `*out_len`. The caller MUST release it with
/// `cuda_oxide_free(*out_ptr, *out_len)`.
///
/// On failure: returns a nonzero code and leaves `*out_ptr`/`*out_len` zeroed.
///
/// Return codes:
///   * `0` -- success.
///   * `1` -- a required pointer argument was NULL.
///   * `2` -- a string argument was not valid UTF-8.
///   * `3` -- the kernel source file does not exist.
///   * `4` -- dependency-rlib resolution failed.
///   * `5` -- in-process codegen failed (the kernel did not yield PTX).
///   * `6` -- a panic was caught at the C ABI boundary. The body is wrapped in
///     `std::panic::catch_unwind`, so a Rust panic is contained and reported as
///     this code instead of unwinding across the FFI boundary (which would abort
///     the host process). NOTE: this does NOT catch a process exit/abort from
///     rustc's own fatal-error handler; see the crate README on broken-kernel
///     behaviour.
///
/// The kernel's dependency rlibs (`cuda_core`/`cuda_host`/`cuda_device`) are
/// resolved from the workspace `target/release/deps` via the best-effort
/// resolver, matching the workspace resolver used by the binary smoke driver.
///
/// # Safety
/// `src_path`/`crate_name`/`crate_version` must be valid NUL-terminated C
/// strings; `arch` must be `NULL` or one. `out_ptr`/`out_len` must be valid,
/// writable pointers.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_oxide_compile(
    src_path: *const c_char,
    crate_name: *const c_char,
    crate_version: *const c_char,
    arch: *const c_char,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    // Contain any panic from the in-process rustc/codegen pipeline at the FFI
    // boundary: an unwind across `extern "C"` aborts the host process. A caught
    // panic is reported as the distinct code 6. (This cannot catch a process
    // exit/abort from rustc's own fatal-error handler -- see the crate README.)
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
        compile_impl(
            src_path,
            crate_name,
            crate_version,
            arch,
            out_ptr,
            out_len,
        )
    }));
    // A caught panic (Err) becomes the distinct code 6.
    result.unwrap_or(6)
}

/// The actual body of [`cuda_oxide_compile`], factored out so the C entry point
/// can wrap it in `catch_unwind`. Same safety contract as the caller.
unsafe fn compile_impl(
    src_path: *const c_char,
    crate_name: *const c_char,
    crate_version: *const c_char,
    arch: *const c_char,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> c_int {
    if src_path.is_null()
        || crate_name.is_null()
        || crate_version.is_null()
        || out_ptr.is_null()
        || out_len.is_null()
    {
        return 1;
    }
    // Initialise outputs.
    unsafe {
        *out_ptr = std::ptr::null_mut();
        *out_len = 0;
    }

    let verbose = std::env::var("CUDA_OXIDE_VERBOSE").is_ok();

    let src = match unsafe { CStr::from_ptr(src_path) }.to_str() {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let crate_name = match unsafe { CStr::from_ptr(crate_name) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 2,
    };
    let crate_version = match unsafe { CStr::from_ptr(crate_version) }.to_str() {
        Ok(s) => s.to_string(),
        Err(_) => return 2,
    };
    let arch = match unsafe { opt_cstr(arch) } {
        Ok(a) => a.map(|s| s.to_string()),
        Err(()) => return 2,
    };

    let kernel_src = PathBuf::from(src);
    if !kernel_src.exists() {
        return 3;
    }

    let dep_rlibs = match compile_core::resolve_workspace_dep_rlibs() {
        Ok(r) => r,
        Err(e) => {
            if verbose {
                eprintln!("cuda_oxide_compile: {e}");
            }
            return 4;
        }
    };

    let req = CompileRequest {
        kernel_src,
        crate_name,
        crate_version,
        dep_rlibs,
        // Per-call, caller-isolated output dir (no shared global temp dir).
        out_dir: unique_out_dir("cuda_oxide_libmode_cdylib_out"),
        arch,
    };

    match compile_to_ptx(&req) {
        Ok(ptx) => {
            // Convert to boxed slice (capacity guaranteed == len).
            let boxed: Box<[u8]> = ptx.into_boxed_slice();
            let len = boxed.len();
            let ptr = Box::into_raw(boxed) as *mut u8;
            unsafe {
                *out_ptr = ptr;
                *out_len = len;
            }
            0
        }
        Err(e) => {
            if verbose {
                eprintln!("cuda_oxide_compile: {e}");
            }
            5
        }
    }
}

/// Free a PTX buffer returned by `cuda_oxide_compile`.
///
/// # Safety
/// `ptr`/`len` must be exactly the pair handed back by a successful
/// `cuda_oxide_compile`, freed at most once. ptr must be the result of
/// `Box::into_raw(Box<[u8]>) as *mut u8`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn cuda_oxide_free(ptr: *mut u8, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    // SAFETY: ptr/len came from Box::into_raw(Box<[u8]>) above.
    unsafe {
        let slice = std::slice::from_raw_parts_mut(ptr, len);
        drop(Box::from_raw(slice as *mut [u8]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;

    // Same `rustc_private` runtime requirements as the bin crate's tests, so
    // this is `#[ignore]`d and must be run explicitly:
    //
    //   cargo build --release -p cuda-core -p cuda-host -p cuda-device
    //   LD_LIBRARY_PATH="$(rustc --print sysroot)/lib" \
    //   RUST_MIN_STACK=16777216 \
    //     cargo test -p cuda-oxide-compiler-cdylib -- --ignored

    /// The C ABI must report a broken kernel as a nonzero return, NOT abort the
    /// host process. A kernel type error makes the in-process rustc raise a
    /// recoverable `FatalError` (caught in `compile_to_ptx`), so `compile_to_ptx`
    /// returns `Err` and `cuda_oxide_compile` returns code 5 (codegen failure).
    /// Reaching the assertion at all also proves the call returned rather than
    /// aborting.
    #[test]
    #[ignore = "requires rustc_private runtime + prebuilt release rlibs; run with --ignored"]
    fn c_abi_broken_kernel_returns_nonzero_not_abort() {
        let src = compile_core::workspace_root()
            .join("crates/cuda-oxide-compiler/tests/fixtures/broken_kernel/src/main.rs");
        let src_c = CString::new(src.to_string_lossy().as_bytes()).unwrap();
        let name_c = CString::new("broken_kernel").unwrap();
        let ver_c = CString::new("0.1.0").unwrap();
        let arch_c = CString::new("sm_80").unwrap();

        let mut out_ptr: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let code = unsafe {
            cuda_oxide_compile(
                src_c.as_ptr(),
                name_c.as_ptr(),
                ver_c.as_ptr(),
                arch_c.as_ptr(),
                &mut out_ptr,
                &mut out_len,
            )
        };

        assert_ne!(code, 0, "broken kernel should yield a nonzero C-ABI code");
        assert!(out_ptr.is_null(), "out_ptr must stay null on failure");
        assert_eq!(out_len, 0, "out_len must stay 0 on failure");
        eprintln!("c_abi broken kernel returned code {code} (no abort)");
    }
}
