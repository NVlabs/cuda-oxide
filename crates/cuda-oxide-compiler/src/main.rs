// SPDX-License-Identifier: Apache-2.0
//
// Thin CLI over the cuda-oxide library-mode compile core.
//
// The in-process Rust-kernel -> PTX compiler lives in the shared
// `compile_core` module (source-included below); this binary is just a smoke
// driver that compiles the bundled `vecadd` example kernel crate to PTX
// in-process and asserts a `.visible .entry`. No `cargo`/kernel-`rustc`
// subprocess is spawned.
//
// REUSE STRATEGY: the backend crate `rustc_codegen_cuda` is a Rust `dylib`.
// Linking that dylib into a binary that also links the `rustc_driver` dylib is
// unsatisfiable -- it doubles the entire `std`/`object`/... graph. Its two
// *device* modules, however, depend only on plain rlibs (mir-importer,
// llvm-export, reserved-oxide-symbols) and rustc_private crates, NOT on
// oxide-artifacts/`object`. So we compile those two source files directly into
// THIS binary via `#[path]`, unchanged. This needs ZERO edits to
// `rustc_codegen_cuda`.
//
// BIN-ONLY: there is deliberately no `src/lib.rs`. A `rustc_private` *library*
// rlib target makes rustc reject the final link ("cannot satisfy dependencies
// so `std` only shows up once") once any plain crates.io rlib (e.g.
// pliron->hashbrown via mir-importer) is in the graph. Keeping everything in
// the binary crate -- the binary crate links the rustc_driver dylib directly,
// which sidesteps that and links cleanly.

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

// Source-include the backend's device pipeline modules, unchanged.
// `device_codegen` refers to `crate::collector`, which resolves because we
// mount `collector` at the crate root here, exactly as the backend's lib.rs
// does.
#[path = "../../rustc-codegen-cuda/src/collector.rs"]
pub mod collector;
#[path = "../../rustc-codegen-cuda/src/device_codegen.rs"]
pub mod device_codegen;

// The SINGLE shared compile core, included by both this bin and the sibling
// cdylib (which `#[path]`s the same file). See `compile_core.rs`.
#[path = "compile_core.rs"]
mod compile_core;

use compile_core::{
    CompileRequest, compile_to_ptx, resolve_workspace_dep_rlibs, unique_out_dir, workspace_root,
};

fn main() {
    // Smoke path: compile the bundled `vecadd` example kernel crate, resolving
    // its host-dep rlibs with the best-effort workspace resolver.
    let kernel_src =
        workspace_root().join("crates/rustc-codegen-cuda/examples/vecadd/src/main.rs");

    let dep_rlibs = match resolve_workspace_dep_rlibs() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("LIBMODE_FAIL: {e}");
            std::process::exit(1);
        }
    };

    let req = CompileRequest {
        kernel_src,
        crate_name: "vecadd".to_string(),
        crate_version: "0.1.0".to_string(),
        dep_rlibs,
        out_dir: unique_out_dir("cuda_oxide_libmode_out"),
        arch: None,
    };

    let ptx = match compile_to_ptx(&req) {
        Ok(ptx) => ptx,
        Err(e) => {
            eprintln!("LIBMODE_FAIL: {e}");
            std::process::exit(1);
        }
    };

    assert!(!ptx.is_empty(), "PTX bytes are empty");
    let text = String::from_utf8_lossy(&ptx);
    let entry_line = match text.lines().find(|l| l.contains(".visible .entry")) {
        Some(l) => l,
        None => {
            eprintln!("LIBMODE_FAIL: PTX has no `.visible .entry`\n{text}");
            std::process::exit(1);
        }
    };

    println!("LIBMODE_OK ptx_bytes={}", ptx.len());
    println!("entry: {}", entry_line.trim());
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests drive `rustc_driver` in-process and so require the full
    // `rustc_private` runtime: a nightly toolchain with `rustc-dev`/`rust-src`,
    // the rustc driver/LLVM shared libs on `LD_LIBRARY_PATH`, a large thread
    // stack, and the `cuda_core`/`cuda_host`/`cuda_device` release rlibs already
    // built. A plain `cargo test --workspace` cannot satisfy this, so every test
    // here is `#[ignore]`d and must be run explicitly.
    //
    // Run recipe:
    //
    //   cargo build --release -p cuda-core -p cuda-host -p cuda-device
    //   LD_LIBRARY_PATH="$(rustc --print sysroot)/lib" \
    //   RUST_MIN_STACK=16777216 \
    //     cargo test -p cuda-oxide-compiler -- --ignored
    //
    // `RUST_MIN_STACK=16777216` is mandatory (LLVM's FPPassManager overflows the
    // default thread stack). The `LD_LIBRARY_PATH` lets the test binary find
    // `librustc_driver-*.so`/`libLLVM-*.so` in the toolchain sysroot.

    #[test]
    #[ignore = "requires rustc_private runtime + prebuilt release rlibs; run with --ignored (see module doc)"]
    fn vecadd_compiles_to_ptx_in_process() {
        let req = CompileRequest {
            kernel_src: workspace_root()
                .join("crates/rustc-codegen-cuda/examples/vecadd/src/main.rs"),
            crate_name: "vecadd".to_string(),
            crate_version: "0.1.0".to_string(),
            dep_rlibs: resolve_workspace_dep_rlibs().expect("resolve dep rlibs"),
            out_dir: unique_out_dir("cuda_oxide_libmode_test_out"),
            arch: None,
        };
        let ptx = compile_to_ptx(&req).expect("in-process PTX compile");
        assert!(!ptx.is_empty(), "PTX bytes are empty");
        let text = String::from_utf8_lossy(&ptx);
        assert!(
            text.contains(".visible .entry"),
            "PTX missing a `.visible .entry`"
        );
        eprintln!("ptx_bytes={}", ptx.len());
    }

    /// Regression-lock the central library-mode claim: compile vecadd in-process
    /// at sm_80 and assert the output matches the committed golden PTX.
    ///
    /// Golden: `tests/golden/vecadd.sm_80.ptx` (pinned to workspace
    /// `rust-toolchain.toml` nightly and the sm_80 target).
    ///
    /// Normalisation: both sides have `//`-to-end-of-line inline comments
    /// stripped before comparison, because LLVM's .ll->PTX lowering emits
    /// cosmetic basic-block label comments (e.g. `// %bb5` vs `// %bb6`) that
    /// can vary across LLVM builds without any instruction-level delta. The
    /// `.visible .entry vecadd(` parameter block is also asserted byte-for-byte
    /// (no normalisation needed there -- it derives from the Rust kernel
    /// signature and is unconditionally stable).
    ///
    /// To (re-)generate the golden:
    ///   CUDA_OXIDE_GENERATE_GOLDEN=1 cargo test -p cuda-oxide-compiler vecadd_golden_parity
    ///
    /// Requires RUST_MIN_STACK=16777216 at run time (LLVM FPPassManager overflows the default stack).
    #[test]
    #[ignore = "requires rustc_private runtime + prebuilt release rlibs; run with --ignored (see module doc)"]
    fn vecadd_golden_parity() {
        let req = CompileRequest {
            kernel_src: workspace_root()
                .join("crates/rustc-codegen-cuda/examples/vecadd/src/main.rs"),
            crate_name: "vecadd".to_string(),
            crate_version: "0.1.0".to_string(),
            dep_rlibs: resolve_workspace_dep_rlibs().expect("resolve dep rlibs"),
            out_dir: unique_out_dir("cuda_oxide_golden_test_out"),
            arch: Some("sm_80".to_string()),
        };
        let ptx = compile_to_ptx(&req).expect("in-process PTX compile at sm_80");
        let text = String::from_utf8_lossy(&ptx);

        let golden_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/golden/vecadd.sm_80.ptx");

        // Generate mode: write the golden file, then return.
        if std::env::var("CUDA_OXIDE_GENERATE_GOLDEN").is_ok() {
            std::fs::create_dir_all(golden_path.parent().unwrap())
                .expect("create tests/golden dir");
            std::fs::write(&golden_path, text.as_bytes()).expect("write golden PTX");
            eprintln!("GOLDEN_WRITTEN: {}", golden_path.display());
            return;
        }

        let golden_bytes = std::fs::read(&golden_path).unwrap_or_else(|_| {
            panic!(
                "golden PTX not found: {}; regenerate with \
                 CUDA_OXIDE_GENERATE_GOLDEN=1 cargo test -p cuda-oxide-compiler \
                 vecadd_golden_parity",
                golden_path.display()
            )
        });
        let golden_text = String::from_utf8_lossy(&golden_bytes);

        // Strip `//`-to-end-of-line inline comments from every line, then
        // trim trailing whitespace on each line.
        let normalise = |s: &str| -> Vec<String> {
            s.lines()
                .map(|line| match line.find("//") {
                    Some(idx) => line[..idx].trim_end().to_string(),
                    None => line.to_string(),
                })
                .collect()
        };

        // Extract the `.visible .entry vecadd(` parameter block (lines from
        // that declaration up to and including the closing `)` line).
        let extract_entry_block = |s: &str| -> String {
            let mut collecting = false;
            let mut lines: Vec<&str> = Vec::new();
            for line in s.lines() {
                if line.contains(".visible .entry vecadd(") {
                    collecting = true;
                }
                if collecting {
                    lines.push(line);
                    if line.trim() == ")" {
                        break;
                    }
                }
            }
            lines.join("\n")
        };

        // 1. PTX must declare sm_80.
        assert!(
            text.contains(".target sm_80"),
            "PTX missing `.target sm_80`"
        );

        // 2. Entry block byte-for-byte match (no normalisation needed).
        let entry_golden = extract_entry_block(&golden_text);
        let entry_got = extract_entry_block(&text);
        assert!(
            !entry_golden.is_empty(),
            "golden PTX has no .visible .entry vecadd( block"
        );
        assert_eq!(
            entry_golden, entry_got,
            ".visible .entry vecadd parameter block mismatch"
        );

        // 3. Full normalised comparison (cosmetic // comments stripped).
        let norm_golden = normalise(&golden_text);
        let norm_got = normalise(&text);
        assert_eq!(
            norm_golden, norm_got,
            "normalised PTX mismatch (after stripping // inline comments)"
        );

        eprintln!("golden_parity OK: ptx_bytes={}", ptx.len());
    }

    /// Compile a deliberately BROKEN kernel crate (a type error inside a
    /// `#[kernel]` fn; fixture at `tests/fixtures/broken_kernel/`) and assert the
    /// failure is reported cleanly as an `Err` from `compile_to_ptx`, rather than
    /// terminating the host process.
    ///
    /// OBSERVED BEHAVIOUR (nightly-2026-04-03): `compile_to_ptx` returns `Err`
    /// cleanly. The in-process `rustc_driver::run_compiler` emits the diagnostics
    /// and reaches `after_analysis` with no successful codegen, so the device
    /// collector finds no functions and `compile_to_ptx` surfaces an `Err`; the
    /// host process is NOT aborted and the call is recoverable. The companion
    /// C-ABI path therefore returns a nonzero code (not an abort).
    ///
    /// Same run requirements as the other tests in this module (see module doc).
    #[test]
    #[ignore = "requires rustc_private runtime + prebuilt release rlibs; run with --ignored (see module doc)"]
    fn broken_kernel_returns_err_not_abort() {
        let req = CompileRequest {
            kernel_src: std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("tests/fixtures/broken_kernel/src/main.rs"),
            crate_name: "broken_kernel".to_string(),
            crate_version: "0.1.0".to_string(),
            dep_rlibs: resolve_workspace_dep_rlibs().expect("resolve dep rlibs"),
            out_dir: unique_out_dir("cuda_oxide_broken_kernel_out"),
            arch: Some("sm_80".to_string()),
        };
        // The crux assertion: a broken kernel yields a recoverable Err and does
        // NOT abort/exit the process. (If the process were terminated by rustc's
        // fatal-error handler, this line would never be reached and the test
        // binary would die with a nonzero signal/exit, which the harness reports
        // as a failure rather than a passing assertion.)
        match compile_to_ptx(&req) {
            Ok(ptx) => panic!(
                "expected Err from a broken kernel, got Ok with {} PTX bytes",
                ptx.len()
            ),
            Err(e) => eprintln!("broken_kernel returned Err cleanly: {e}"),
        }
    }
}
