// SPDX-License-Identifier: Apache-2.0
//
// Canonical usage demo for the cuda-oxide library-mode cdylib.
//
// This binary models what a runtime-JIT host (e.g. CubeCL or Burn) would do:
// load the cuda-oxide in-process compiler at startup via `dlopen`
// (`libloading`), then call the C ABI to JIT Rust device-kernel crates to PTX
// -- one library load amortised across many compiles, no `cargo`/`rustc`
// subprocess.
//
// No `#![feature(rustc_private)]`, no `rustc_driver` link dependency: this is
// a standard Rust binary that links only against libc and libloading. The
// compiler logic lives entirely inside the cdylib (`cuda-oxide-compiler-cdylib`).
//
// # Runtime requirements (must be set before launch)
//
//   LD_LIBRARY_PATH=$(rustc --print sysroot)/lib
//       Required so the dynamic linker finds `librustc_driver-*.so` and
//       `libLLVM-*.so` inside the cdylib.
//
//   GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
//       `rustc_driver` uses initial-exec TLS internally; glibc's dynamic TLS
//       allocator must be given enough slack to accommodate it at dlopen time.
//       Without this knob the load fails with a TLS-allocation error on glibc >= 2.34.
//
// # C ABI called here
//
//   cuda_oxide_compile(src_path, crate_name, crate_version, arch,
//                      out_ptr: *mut *mut u8, out_len: *mut usize) -> c_int
//       Compiles the Rust device-kernel crate at `src_path` to PTX in-process.
//       Pass explicit crate identity (`crate_name`, `crate_version`) so the
//       `#[cuda_module]`/`#[kernel]` macros derive the right module symbols.
//       `arch` is NUL or a NUL-terminated string like "sm_80"; NULL defers to
//       the pipeline's auto-detection. Returns 0 on success; nonzero on error.
//       The caller must release the buffer with `cuda_oxide_free`.
//
//   cuda_oxide_free(ptr: *mut u8, len: usize)
//       Releases a buffer returned by `cuda_oxide_compile`.
//
// # Usage
//   libmode-consumer <path-to-cdylib> <path-to-kernel-src> [n_repeats]
//
// Prints the PTX byte count and `.visible .entry` line for the first compile;
// with `n_repeats > 1` also reports per-compile latency statistics (steady-state
// median excludes the first cold compile).

use std::ffi::CString;
use std::os::raw::{c_char, c_int};
use std::time::Instant;

// extern "C" fn cuda_oxide_compile(src, crate_name, crate_version, arch,
//                                   out_ptr: *mut *mut u8, out_len: *mut usize) -> c_int
// `arch` is nullable: NULL defers to the pipeline's auto-detection.
type CompileFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *const c_char,
    *mut *mut u8,
    *mut usize,
) -> c_int;
// extern "C" fn cuda_oxide_free(ptr: *mut u8, len: usize)
type FreeFn = unsafe extern "C" fn(*mut u8, usize);

fn main() {
    let mut args = std::env::args().skip(1);
    let lib_path = args
        .next()
        .expect("usage: libmode-consumer <cdylib> <kernel-src> [n_repeats]");
    let kernel_src = args
        .next()
        .expect("usage: libmode-consumer <cdylib> <kernel-src> [n_repeats]");
    let n_repeats: usize = args.next().map(|s| s.parse().expect("n_repeats")).unwrap_or(1);

    // ---- one-time library load (the dlopen) -------------------------------
    let t_load = Instant::now();
    let lib = unsafe { libloading::Library::new(&lib_path) }
        .unwrap_or_else(|e| panic!("dlopen {lib_path} failed: {e}"));
    let compile: libloading::Symbol<CompileFn> =
        unsafe { lib.get(b"cuda_oxide_compile\0") }.expect("symbol cuda_oxide_compile");
    let free: libloading::Symbol<FreeFn> =
        unsafe { lib.get(b"cuda_oxide_free\0") }.expect("symbol cuda_oxide_free");
    let load_us = t_load.elapsed().as_micros();
    println!("LOADED cdylib in {load_us} us (one-time)");

    let c_src = CString::new(kernel_src.as_str()).expect("kernel src has NUL");
    // Explicit crate identity for the vecadd example (was hardwired in the
    // cdylib). `arch = NULL` defers to the pipeline's auto-detection (sm_80),
    // matching the byte-for-byte parity baseline.
    let c_crate_name = CString::new("vecadd").expect("crate name has NUL");
    let c_crate_version = CString::new("0.1.0").expect("crate version has NUL");
    let arch_ptr: *const c_char = std::ptr::null();

    let mut first_ptx_len = 0usize;
    let mut first_entry = String::new();
    let mut per_compile_us: Vec<u128> = Vec::with_capacity(n_repeats);

    for i in 0..n_repeats {
        let mut out_ptr: *mut u8 = std::ptr::null_mut();
        let mut out_len: usize = 0;

        let t = Instant::now();
        let rc: c_int = unsafe {
            compile(
                c_src.as_ptr() as *const c_char,
                c_crate_name.as_ptr() as *const c_char,
                c_crate_version.as_ptr() as *const c_char,
                arch_ptr,
                &mut out_ptr,
                &mut out_len,
            )
        };
        let us = t.elapsed().as_micros();
        per_compile_us.push(us);

        if rc != 0 {
            panic!("cuda_oxide_compile returned {rc} on iteration {i}");
        }
        assert!(!out_ptr.is_null() && out_len > 0, "empty PTX buffer");

        // Copy PTX out into owned (normal) Rust memory, then free via the lib.
        let ptx: Vec<u8> =
            unsafe { std::slice::from_raw_parts(out_ptr, out_len) }.to_vec();
        unsafe { free(out_ptr, out_len) };

        if i == 0 {
            first_ptx_len = ptx.len();
            let text = String::from_utf8_lossy(&ptx);
            first_entry = text
                .lines()
                .find(|l| l.contains(".visible .entry"))
                .map(|l| l.trim().to_string())
                .unwrap_or_else(|| "<no .visible .entry found>".to_string());
            // Persist the in-process PTX for the parity comparison (Part B).
            let dump = std::env::temp_dir().join("cuda_oxide_libmode_consumer.ptx");
            std::fs::write(&dump, &ptx).expect("write PTX dump");
            println!("DUMP {}", dump.display());
        }
    }

    println!("CONSUMER_OK ptx_bytes={first_ptx_len}");
    println!("entry: {first_entry}");

    if n_repeats >= 1 {
        let mut sorted = per_compile_us.clone();
        sorted.sort_unstable();
        let median = sorted[sorted.len() / 2];
        let min = *sorted.first().unwrap();
        let max = *sorted.last().unwrap();
        // Steady-state median EXCLUDES the first (cold) compile when we have
        // enough samples.
        let steady_median = if sorted.len() > 1 {
            let warm: Vec<u128> = per_compile_us[1..].to_vec();
            let mut w = warm.clone();
            w.sort_unstable();
            w[w.len() / 2]
        } else {
            median
        };
        println!(
            "LATENCY n={n_repeats} load_us={load_us} \
             compile_us[first={} median_all={median} min={min} max={max} steady_median={steady_median}]",
            per_compile_us[0]
        );
    }
}
