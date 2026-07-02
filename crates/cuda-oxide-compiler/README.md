# cuda-oxide-compiler

In-process Rust device-kernel to PTX compiler for the cuda-oxide library-mode
feature (NVlabs/cuda-oxide#96).

## What library mode is

The standard cuda-oxide build path spawns a `cargo oxide build` subprocess per
kernel, which in turn spawns a per-kernel `rustc` process. Library mode
eliminates those subprocesses: `rustc`'s front-end (parse, type-check,
monomorphisation, MIR lowering) runs inside the calling process via
`rustc_driver::run_compiler`, and the cuda-oxide device pipeline
(`collect_device_functions` + `generate_device_code`) is driven directly from
the `after_analysis` callback. The only remaining subprocesses are `llc` and
`opt` inside `run_pipeline` (the .ll to PTX step), which are short-lived and
not per-kernel-start-up costs.

The result: one library load, then many in-process compiles -- no `cargo` or
kernel-`rustc` child per kernel. Measured steady-state latency is about 158 ms
per compile vs. about 445 ms for the subprocess path (roughly 2.8 times faster,
RTX 5060 Laptop, warm incremental cache).

## Rust API (`compile_to_ptx`)

```rust
use cuda_oxide_compiler::{CompileRequest, compile_to_ptx, resolve_workspace_dep_rlibs, unique_out_dir};

let req = CompileRequest {
    kernel_src: PathBuf::from("path/to/kernel/src/lib.rs"),
    crate_name:    "my_kernel".to_string(),
    crate_version: "0.1.0".to_string(),
    dep_rlibs: resolve_workspace_dep_rlibs()?,   // or supply explicit paths
    out_dir:   unique_out_dir("my_kernel_out"),
    arch:      Some("sm_80".to_string()),         // None = auto-detect
};
let ptx: Vec<u8> = compile_to_ptx(&req)?;
```

`CompileRequest` fields:

| Field | Type | Description |
|---|---|---|
| `kernel_src` | `PathBuf` | Path to the kernel crate root `.rs` |
| `crate_name` | `String` | Crate name; feeds `CARGO_PKG_NAME` read by `#[cuda_module]`/`#[kernel]` |
| `crate_version` | `String` | Crate version; feeds `CARGO_PKG_VERSION` |
| `dep_rlibs` | `Vec<PathBuf>` | Explicit `lib<name>-<hash>.rlib` paths for `cuda_core`/`cuda_host`/`cuda_device` and transitives |
| `out_dir` | `PathBuf` | Per-call, caller-isolated directory for `.ll`/`.ptx` artifacts |
| `arch` | `Option<String>` | Device target, e.g. `"sm_80"`; `None` defers to pipeline auto-detection |

`resolve_workspace_dep_rlibs()` is a best-effort heuristic (most-recent-mtime
rlib in `target/release/deps`). For production callers, supply `dep_rlibs`
explicitly from the build system.

## C ABI (cdylib -- `cuda-oxide-compiler-cdylib`)

The companion crate `cuda-oxide-compiler-cdylib` packages the same logic as a
`cdylib` with a C-ABI surface for `dlopen`/`libloading` consumers:

```c
// Compile kernel at src_path to PTX in-process. Returns 0 on success.
// arch: NUL or a NUL-terminated string like "sm_80"; NULL = auto-detect.
// On success: *out_ptr/out_len are set; caller must free with cuda_oxide_free.
int cuda_oxide_compile(
    const char *src_path,
    const char *crate_name,
    const char *crate_version,
    const char *arch,
    uint8_t   **out_ptr,
    size_t     *out_len
);

// Release a buffer returned by cuda_oxide_compile.
void cuda_oxide_free(uint8_t *ptr, size_t len);
```

`cuda_oxide_compile` return codes: `0` success; `1` a required pointer was
NULL; `2` a string argument was not valid UTF-8; `3` the kernel source file does
not exist; `4` dependency-rlib resolution failed; `5` codegen failed (including a
kernel that does not compile); `6` a panic was caught at the C ABI boundary. The
body is wrapped in `std::panic::catch_unwind`, so a Rust panic is contained and
reported as code 6 instead of unwinding across the FFI boundary and aborting the
host process.

See `crates/libmode-consumer/` for the canonical Rust `dlopen` usage demo.

## Broken-kernel behaviour

A kernel that fails rustc's own front-end (e.g. a type error in a `#[kernel]`
fn) is handled cleanly: `compile_to_ptx` returns `Err`, and the cdylib's
`cuda_oxide_compile` returns code 5. The host process is NOT terminated.

Internally rustc aborts such a compilation by unwinding with a sentinel
`FatalError` rather than returning; `compile_to_ptx` wraps `run_compiler` in
`catch_fatal_errors` (the same hook rustc itself uses) to convert that unwind
into a recoverable error. A genuine internal compiler error (any non-`FatalError`
panic) is re-raised; at the C ABI the outer `catch_unwind` contains it as code 6.

## Device-only crate constraint

The kernel crate compiled by `compile_to_ptx` must be a **device-only** crate:
it must not import any host crate that pulls in `oxide-artifacts` or the
`object` crate. Doing so doubles the dependency graph (`std`/`object` appear
twice) and causes a link error. The permitted dependencies are `cuda_core`,
`cuda_host`, and `cuda_device` (and any crate they re-export), exactly the
set produced by `resolve_workspace_dep_rlibs`.

## Runtime requirements

### `LD_LIBRARY_PATH`

```
LD_LIBRARY_PATH=$(rustc --print sysroot)/lib
```

The process (or the process loading the cdylib) must find `librustc_driver-*.so`
and `libLLVM-*.so` at runtime. These live in the toolchain sysroot `lib/`
directory, which is not on the system library path by default.

### `GLIBC_TUNABLES` (initial-exec TLS)

```
GLIBC_TUNABLES=glibc.rtld.optional_static_tls=2097152
```

`rustc_driver` uses initial-exec TLS internally. When loaded as a `dlopen`
target (cdylib) on glibc >= 2.34, the dynamic linker must be given enough
static TLS slack to accommodate this at load time. Without this knob the load
fails with a TLS-allocation error. The value `2097152` (2 MiB) is sufficient
for the current toolchain; it may need adjustment on future nightlies.

This knob is only required when loading the cdylib via `dlopen`; it is not
needed when using the `cuda-oxide-compiler` binary directly (which links
`rustc_driver` at start-up, before initial-exec TLS is locked).

## `llc`/`opt` remain subprocesses

The `.ll` to PTX step inside `run_pipeline` spawns `llc` and `opt` as child
processes. These are short-lived per-compile, not per-kernel start-up costs,
and are inherited from the existing cuda-oxide device pipeline. Eliminating
them would require embedding the LLVM NVPTX back-end directly and is out of
scope for this crate.

## Testing

The tests in this crate (and in `cuda-oxide-compiler-cdylib`) drive
`rustc_driver` in-process, so they need the full `rustc_private` runtime: a
nightly toolchain with `rustc-dev`/`rust-src`, the rustc-driver/LLVM shared libs
on `LD_LIBRARY_PATH`, a large thread stack, and the `cuda_core`/`cuda_host`/
`cuda_device` release rlibs already built. A plain `cargo test --workspace`
cannot satisfy this, so every such test is marked `#[ignore]` and is skipped by
the default run. Run them explicitly:

```
cargo build --release -p cuda-core -p cuda-host -p cuda-device
LD_LIBRARY_PATH="$(rustc --print sysroot)/lib" \
RUST_MIN_STACK=16777216 \
  cargo test -p cuda-oxide-compiler -p cuda-oxide-compiler-cdylib -- --ignored
```

`RUST_MIN_STACK=16777216` is mandatory (LLVM's FPPassManager overflows the
default thread stack). The ignored tests cover: the in-process `vecadd` compile,
golden-PTX parity at sm_80, and the broken-kernel path (a kernel type error
returns `Err`/a nonzero C-ABI code, not a process abort).

## Known limitations and follow-ups

- **Compiles must be serialised.** `compile_to_ptx` mutates process-global
  environment variables (`CARGO_PKG_NAME`, `CARGO_PKG_VERSION`,
  `CUDA_OXIDE_TARGET`) that the in-process `rustc` and proc macros read via
  `std::env::var`. Concurrent calls from multiple threads produce races.
  A future revision should replace the env-var channel with a thread-local or
  per-compile mechanism.

- **`sysroot()` and `workspace_root()` panic on misconfiguration.** Both
  functions panic if the sysroot subprocess fails or the manifest-dir layout is
  unexpected. They should return `Result` to give callers a recoverable error
  path.

- **`resolve_workspace_dep_rlibs` is best-effort.** It picks the most recently
  modified rlib by mtime, which is fragile when multiple builds exist in the
  same `deps/` directory. Prefer explicit `dep_rlibs` for any non-trivial
  caller.

- **Toolchain pin.** The golden PTX in `tests/golden/vecadd.sm_80.ptx` is
  pinned to the workspace `rust-toolchain.toml` nightly
  (`nightly-2026-04-03`). Updating the toolchain requires regenerating the
  golden with `CUDA_OXIDE_GENERATE_GOLDEN=1 cargo test -p cuda-oxide-compiler
  vecadd_golden_parity`.

- **Test stack size requirement.** Running the (ignored) test suite requires `RUST_MIN_STACK=16777216` (see Testing above); without it LLVM overflows the default thread stack.
