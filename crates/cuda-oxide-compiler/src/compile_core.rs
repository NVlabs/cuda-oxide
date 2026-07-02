// SPDX-License-Identifier: Apache-2.0
//
// Shared in-process compile core for the cuda-oxide library-mode compiler.
//
// This is the SINGLE implementation of the in-process Rust-kernel -> PTX
// compiler. Both the bin (`cuda-oxide-compiler/src/main.rs`) and the cdylib
// (`cuda-oxide-compiler-cdylib/src/lib.rs`) source-include THIS file via
// `#[path]`, so the driver/`Callbacks`, the rustc-arg builder, sysroot
// handling, dep-rlib resolution and output handling cannot drift between the
// two targets.
//
// It depends on two items mounted at the including crate's root, exactly as the
// backend's own `lib.rs` mounts them:
//   * `crate::collector`        (rustc-codegen-cuda/src/collector.rs)
//   * `crate::device_codegen`   (rustc-codegen-cuda/src/device_codegen.rs)
// and on the `rustc_*` private crates the including crate brings in via
// `extern crate` (visible here through the extern prelude, the same way
// `collector.rs`/`device_codegen.rs` see them).
//
// The codegen *backend* (`-Z codegen-backend=`) is intentionally NOT used: we
// stop after analysis and call the device pipeline directly, so host codegen
// and linking never run. The only subprocess is `llc`/`opt` (.ll -> PTX) inside
// `run_pipeline`; cargo and the kernel-`rustc` are never spawned.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::device_codegen::{self, DeviceCodegenConfig};
use rustc_driver::{Callbacks, Compilation};
use rustc_interface::interface::Compiler;
use rustc_middle::ty::TyCtxt;

// --------------------------------------------------------------------------
// Public API
// --------------------------------------------------------------------------

/// Error returned by [`compile_to_ptx`] and the helpers.
#[derive(Debug, Clone)]
pub struct CompileError {
    message: String,
}

impl CompileError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CompileError {}

/// A single, caller-isolated request to compile one Rust device-kernel crate to
/// PTX in-process.
///
/// `dep_rlibs` is the primary, explicit dependency channel: each entry is a
/// path to a `lib<name>-<hash>.rlib` the kernel crate depends on (the
/// `cuda_core`/`cuda_host`/`cuda_device` rlibs and any transitive ones). The
/// `--extern <name>=<path>` name is derived from the rlib filename, and a
/// `-L dependency=<dir>` is added for every directory the rlibs live in.
pub struct CompileRequest {
    /// Path to the kernel crate root `.rs` (e.g. the crate's `main.rs`/`lib.rs`).
    pub kernel_src: PathBuf,
    /// Kernel crate name (was hardwired "vecadd"). Feeds `CARGO_PKG_NAME`, which
    /// the `#[cuda_module]`/`#[kernel]` macros read to derive the embedded
    /// module symbol, and names the emitted `<name>.ptx`.
    pub crate_name: String,
    /// Kernel crate version (was hardwired "0.1.0"). Feeds `CARGO_PKG_VERSION`.
    pub crate_version: String,
    /// Explicit dependency rlib paths. Primary API; prefer this over the
    /// best-effort [`resolve_workspace_dep_rlibs`] resolver.
    pub dep_rlibs: Vec<PathBuf>,
    /// Per-call, caller-isolated output directory for the `.ll`/`.ptx`
    /// artifacts. No shared global temp dir across calls.
    pub out_dir: PathBuf,
    /// Device target, e.g. "sm_80". `None` defers to the pipeline's
    /// auto-detection (or any externally set `CUDA_OXIDE_TARGET`).
    pub arch: Option<String>,
}

/// Compile a Rust device-kernel crate to PTX bytes, fully in-process.
///
/// No `cargo`/kernel-`rustc` subprocess is spawned; rustc's front-end runs in
/// THIS process via `rustc_driver::run_compiler`, and the device pipeline is
/// driven from `after_analysis`. Returns the PTX bytes on success.
///
/// A kernel that fails to compile (e.g. a type error in a `#[kernel]` fn) is
/// reported as a recoverable [`CompileError`]: rustc aborts such a compilation
/// by unwinding with a sentinel `FatalError`, which is caught here (via
/// `catch_fatal_errors`) and converted into `Err`. The host process is NOT
/// terminated.
pub fn compile_to_ptx(req: &CompileRequest) -> Result<Vec<u8>, CompileError> {
    std::fs::create_dir_all(&req.out_dir)
        .map_err(|e| CompileError::new(format!("failed to create output dir: {e}")))?;

    let args = build_rustc_args(req)?;
    let verbose = std::env::var("CUDA_OXIDE_VERBOSE").is_ok();

    // SAFETY / why env (and not rustc args): the `#[cuda_module]`/`#[kernel]`
    // proc macros read `CARGO_PKG_NAME`/`CARGO_PKG_VERSION` via raw
    // `std::env::var` at expansion time (see `cuda-macros`), and the device
    // pipeline reads the target arch from `CUDA_OXIDE_TARGET`. Because rustc
    // runs IN THIS process, all three are read from THIS process's environment.
    // rustc's `--env-set` only feeds `proc_macro::tracked_env::var`/`env!`, NOT
    // the raw `std::env::var` these consumers use, so the arg-vector route
    // cannot supply them. `compile_to_ptx` mutates process-global environment
    // variables and MUST NOT be called concurrently; callers must serialise
    // compiles.
    unsafe {
        std::env::set_var("CARGO_PKG_NAME", &req.crate_name);
        std::env::set_var("CARGO_PKG_VERSION", &req.crate_version);
        match req.arch.as_deref() {
            Some(arch) => std::env::set_var("CUDA_OXIDE_TARGET", arch),
            None => std::env::remove_var("CUDA_OXIDE_TARGET"),
        }
    }

    let mut cb = DeviceCallbacks {
        output_dir: req.out_dir.clone(),
        output_name: req.crate_name.clone(),
        verbose,
        ptx: None,
        error: None,
    };

    // Drive rustc in-process, wrapped in `catch_fatal_errors`. The whole
    // front-end (parse -> analysis) runs here; no rustc/cargo child is spawned
    // for the kernel build.
    //
    // A KERNEL COMPILE ERROR (e.g. a type error in a `#[kernel]` fn) makes rustc
    // ABORT the compilation by unwinding with a sentinel `FatalErrorMarker`
    // panic. That is neither a clean return nor a `process::exit`/abort -- it is
    // a recoverable unwind, and `catch_fatal_errors` is exactly the hook rustc
    // itself uses to turn it back into a `Result`. Without this wrapper the
    // unwind would propagate out of `compile_to_ptx` as a panic (aborting a C
    // consumer across the FFI boundary). With it, a broken kernel becomes a
    // clean `Err` here (and a nonzero return at the cdylib's C ABI). Note: a
    // genuine ICE (any non-`FatalError` panic) is re-raised by
    // `catch_fatal_errors`; the cdylib's outer `catch_unwind` contains that.
    let run = rustc_span::fatal_error::catch_fatal_errors(|| {
        rustc_driver::run_compiler(&args, &mut cb);
    });

    // The PTX bytes are already read back into `cb.ptx` by `generate_device_code`
    // (in `after_analysis`), so the per-call output dir and its `.ll`/`.ptx`
    // artifacts are no longer needed. Remove it best-effort; a failure to clean
    // up the temp dir must not affect the compile result.
    let _ = std::fs::remove_dir_all(&req.out_dir);

    if run.is_err() {
        // rustc aborted on a fatal (compile) error. Surface any specific message
        // the callback recorded; otherwise report the fatal abort generically.
        return Err(CompileError::new(cb.error.unwrap_or_else(|| {
            "kernel compilation failed: rustc reported errors (see diagnostics above)".to_string()
        })));
    }

    match (cb.ptx, cb.error) {
        (Some(ptx), _) => Ok(ptx),
        (None, Some(err)) => Err(CompileError::new(err)),
        (None, None) => Err(CompileError::new(
            "compilation finished without reaching device codegen",
        )),
    }
}

/// Workspace root = two levels up from the including crate's manifest dir.
///
/// Both including crates live at `crates/<name>/`, so this resolves to the
/// cuda-oxide workspace root for either of them.
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

/// Best-effort resolver for the example/test convenience path: find the
/// `cuda_core`/`cuda_host`/`cuda_device` rlibs in the workspace
/// `target/release/deps` by most-recent mtime.
///
/// This is the fragile heuristic the explicit `CompileRequest::dep_rlibs`
/// replaces for real callers; it exists only so the bundled `vecadd` smoke path
/// and tests need not hardcode hashes. Build the deps first with:
/// `cargo build --release -p cuda-core -p cuda-host -p cuda-device`.
pub fn resolve_workspace_dep_rlibs() -> Result<Vec<PathBuf>, CompileError> {
    let deps_dir = workspace_root().join("target/release/deps");
    Ok(vec![
        find_rlib(&deps_dir, "cuda_core")?,
        find_rlib(&deps_dir, "cuda_host")?,
        find_rlib(&deps_dir, "cuda_device")?,
    ])
}

/// A unique, per-call output directory under the system temp dir, isolated from
/// any other concurrent or prior call.
pub fn unique_out_dir(prefix: &str) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    std::env::temp_dir().join(format!("{prefix}-{pid}-{nanos}-{n}"))
}

// --------------------------------------------------------------------------
// Internals
// --------------------------------------------------------------------------

/// Callback state: collects PTX bytes (or an error string) out of the driver.
struct DeviceCallbacks {
    output_dir: PathBuf,
    output_name: String,
    verbose: bool,
    ptx: Option<Vec<u8>>,
    error: Option<String>,
}

impl Callbacks for DeviceCallbacks {
    fn after_analysis(&mut self, _compiler: &Compiler, tcx: TyCtxt<'_>) -> Compilation {
        // Mirror CudaCodegenBackend::codegen_crate, but driven from
        // after_analysis instead of the codegen phase.
        //
        // 1. Partition mono items so the collector can walk the call graph from
        //    each kernel entry point.
        let mono_partitions = tcx.collect_and_partition_mono_items(());

        // 2. Collect device-reachable functions (kernels + callees) and externs.
        let collection =
            crate::collector::collect_device_functions(tcx, mono_partitions.codegen_units, self.verbose);

        if collection.functions.is_empty() {
            self.error = Some(
                "no device functions found (crate has no #[kernel]/#[device] items)".to_string(),
            );
            return Compilation::Stop;
        }

        // 3. Run the existing MIR -> PTX pipeline into our output dir.
        let device_config = DeviceCodegenConfig {
            output_dir: self.output_dir.clone(),
            output_name: self.output_name.clone(),
            verbose: self.verbose,
            dump_rustc_mir: false,
            dump_mir_dialect: false,
            dump_llvm_dialect: false,
        };

        match device_codegen::generate_device_code(
            tcx,
            &collection.functions,
            &collection.device_externs,
            &device_config,
        ) {
            Ok(result) => {
                // `generate_device_code` already read the artifact back as
                // bytes. Prefer the embeddable artifact bytes; fall back to
                // ptx_content.
                if let Some(artifact) = result.artifact {
                    self.ptx = Some(artifact.bytes);
                } else if let Some(ptx) = result.ptx_content {
                    self.ptx = Some(ptx.into_bytes());
                } else {
                    self.error = Some("device codegen produced no PTX artifact".to_string());
                }
            }
            Err(e) => {
                self.error = Some(format!("device codegen failed: {e}"));
            }
        }

        // We have the PTX (or an error); never proceed to host codegen.
        Compilation::Stop
    }
}

/// Resolve the rustc sysroot ONCE per process and cache it. `rustc --print
/// sysroot` is a subprocess, but it does NOT compile the kernel -- it only
/// prints a path -- and it runs at most once thanks to the `OnceLock`. Honours
/// a `CUDA_OXIDE_SYSROOT` override when set.
fn sysroot() -> &'static str {
    static SYSROOT: OnceLock<String> = OnceLock::new();
    SYSROOT.get_or_init(|| {
        if let Ok(s) = std::env::var("CUDA_OXIDE_SYSROOT") {
            return s;
        }
        let out = std::process::Command::new("rustc")
            .args(["--print", "sysroot"])
            .output()
            .expect("run rustc --print sysroot");
        String::from_utf8(out.stdout)
            .expect("utf8 sysroot")
            .trim()
            .to_string()
    })
}

/// Find the most recently modified `lib<name>-*.rlib` in `deps_dir`.
fn find_rlib(deps_dir: &Path, name: &str) -> Result<PathBuf, CompileError> {
    let prefix = format!("lib{name}-");
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let rd = std::fs::read_dir(deps_dir)
        .map_err(|e| CompileError::new(format!("read deps dir {}: {e}", deps_dir.display())))?;
    for entry in rd {
        let entry = entry.map_err(|e| CompileError::new(format!("dir entry: {e}")))?;
        let fname = entry.file_name();
        let fname = fname.to_string_lossy();
        if fname.starts_with(&prefix) && fname.ends_with(".rlib") {
            let mtime = entry
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            if best.as_ref().map(|(t, _)| mtime > *t).unwrap_or(true) {
                best = Some((mtime, entry.path()));
            }
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        CompileError::new(format!(
            "rlib for `{name}` not found in {} -- build deps first: \
             cargo build --release -p cuda-core -p cuda-host -p cuda-device",
            deps_dir.display()
        ))
    })
}

/// Derive the `--extern` crate name from an rlib path `lib<name>-<hash>.rlib`.
fn extern_name_from_rlib(rlib: &Path) -> Result<String, CompileError> {
    let fname = rlib
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| CompileError::new(format!("rlib path has no file name: {}", rlib.display())))?;
    let stem = fname
        .strip_prefix("lib")
        .and_then(|s| s.strip_suffix(".rlib"))
        .ok_or_else(|| {
            CompileError::new(format!(
                "not a `lib<name>-<hash>.rlib` rlib: {}",
                rlib.display()
            ))
        })?;
    // `<name>-<hash>`: the crate name is everything before the final `-`.
    let name = match stem.rsplit_once('-') {
        Some((name, _hash)) => name,
        None => stem,
    };
    Ok(name.to_string())
}

/// Build the rustc arg vector for the kernel crate described by `req`.
///
/// Equivalent to cuda-oxide's release kernel build, MINUS `-Z codegen-backend=`
/// (we call the pipeline directly from `after_analysis`). `--extern`/`-L` are
/// derived from the explicit `dep_rlibs`.
fn build_rustc_args(req: &CompileRequest) -> Result<Vec<String>, CompileError> {
    if !req.kernel_src.exists() {
        return Err(CompileError::new(format!(
            "kernel source missing: {}",
            req.kernel_src.display()
        )));
    }

    let mut args = vec![
        "rustc".to_string(),
        req.kernel_src.to_string_lossy().into_owned(),
        "--edition".to_string(),
        "2021".to_string(),
        "--crate-type".to_string(),
        "lib".to_string(),
        "-Copt-level=3".to_string(),
        "-Cdebug-assertions=off".to_string(),
        "-Zmir-enable-passes=-JumpThreading".to_string(),
        "-Csymbol-mangling-version=v0".to_string(),
        format!("--sysroot={}", sysroot()),
    ];

    // One `--extern name=path` per rlib, plus one `-L dependency=dir` per unique
    // directory the rlibs live in (so transitive deps resolve).
    let mut dep_dirs: Vec<PathBuf> = Vec::new();
    for rlib in &req.dep_rlibs {
        let name = extern_name_from_rlib(rlib)?;
        args.push("--extern".to_string());
        args.push(format!("{name}={}", rlib.display()));
        if let Some(parent) = rlib.parent() {
            let parent = parent.to_path_buf();
            if !dep_dirs.contains(&parent) {
                dep_dirs.push(parent);
            }
        }
    }
    for dir in dep_dirs {
        args.push("-L".to_string());
        args.push(format!("dependency={}", dir.display()));
    }

    args.push("--out-dir".to_string());
    args.push(req.out_dir.to_string_lossy().into_owned());

    Ok(args)
}
