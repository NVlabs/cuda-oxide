# cargo-oxide

Cargo subcommand for building and running Rust GPU programs with cuda-oxide.

Replaces the previous `xtask` pattern with a proper cargo subcommand that works both inside the cuda-oxide repo (for developers) and externally (for users who `cargo install`).

## Installation

**Internal developers** (inside the cuda-oxide repo): no installation needed. The workspace alias makes `cargo oxide` work immediately.

**External users**:

Install with the project's pinned nightly toolchain:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend if it's not already available.

## Usage

```bash
cargo oxide new my_project          # scaffold a new cuda-oxide project
cargo oxide new my_project --async  # scaffold with async template (tokio + cuda-async)
cargo oxide run vecadd              # build + run an example
cargo oxide build vecadd            # compile only (no run)
cargo oxide emit-ltoir vecadd --arch sm_100  # device code -> .ltoir (Tile/SIMT interop)
cargo oxide build -- -p my_app      # arbitrary cargo build through cuda-oxide
cargo oxide test                    # cargo test with cuda-oxide defaults
cargo oxide test -- -p my_app       # arbitrary cargo test through cuda-oxide
cargo oxide pipeline vecadd         # verbose pipeline dump
                                    # (MIR -> dialect-mir -> LLVM dialect -> LLVM IR -> PTX)
cargo oxide sanitize vecadd         # build + run under NVIDIA Compute Sanitizer
cargo oxide debug vecadd --tui      # build + launch cuda-gdb
cargo oxide fmt                     # format all crates
cargo oxide fmt --check             # check formatting
cargo oxide doctor                  # validate environment
cargo oxide setup                   # explicitly build the codegen backend
```

### Flags

| Flag                         | Applies to                       | Description                                     |
|------------------------------|----------------------------------|-------------------------------------------------|
| `--emit-nvvm-ir`             | run, build, pipeline             | Generate NVVM IR for libNVVM                    |
| `--arch <sm_XX>`             | run, sanitize, build, test, pipeline, emit-ltoir | Target architecture override |
| `--features <F>`             | run, sanitize, debug, build, build passthrough, emit-ltoir | Comma-separated cargo features to enable |
| `--bin <NAME>`               | run, sanitize, debug                | Specific binary target to build/run      |
| `--tool <T>`                 | sanitize                         | Compute Sanitizer tool: `memcheck`, `racecheck`, `initcheck`, or `synccheck` |
| `-o, --output <P>`           | emit-ltoir                       | Output path for the `.ltoir` artifact           |
| `--cargo-target-dir <PATH>`  | build/test passthrough           | Cargo target directory                          |
| `--device-codegen-crate <LIST>` | build/test passthrough        | Comma-separated device owner crate filter       |
| `--device-cfg <NAME>`        | build/test passthrough           | Append `--cfg NAME` to rustflags                |
| `-v, --verbose`              | run, sanitize, build, test, emit-ltoir | Show detailed compilation output           |
| `--no-fmad`                  | run, sanitize, build, emit-ltoir, pipeline | Keep ordinary multiply and add/subtract operations separate |
| `--async`                    | new                              | Use the async template                          |
| `--cgdb`                     | debug                            | Use cgdb instead of cuda-gdb                    |
| `--tui`                      | debug                            | Use GDB's TUI interface                         |
| `--check`                    | fmt                              | Check formatting only                           |

`--arch` is required for `emit-ltoir` and explicit NVVM IR output because those
artifacts are architecture-specific. Without an override, `run` detects the
local GPU, while `build` and `pipeline` use the compiler's feature-based target
so they remain useful for cross-compilation.

`--no-fmad` (or `CUDA_OXIDE_NO_FMA=1`) disables implicit contraction of an
ordinary multiply followed by an add or subtract, so the two operations round
separately. Explicit fused operations such as `f32::mul_add` remain fused.
For NVVM IR and LTOIR, cuda-oxide records the policy in matching `.options`
and versioned `.target` files and passes `-fma=0` through libNVVM and
nvJitLink. Keep both sidecars with the artifact when another build system
consumes it.

## Commands

### `cargo oxide run <example>`

Builds the codegen backend, compiles the example with the custom backend, and runs it. This is the primary command for day-to-day development.

When neither `--arch` nor `CUDA_OXIDE_TARGET` is set, `run` detects the
compute capability of CUDA device 0 and targets that architecture so the
generated PTX can load on the local GPU. Use `--arch <sm_XXX>` or
`CUDA_OXIDE_TARGET=<sm_XXX>` to override this for a specific device or
cross-target workflow.

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol
cargo oxide run device_ffi_test --emit-nvvm-ir --arch sm_120
cargo oxide run cutile_inter_kernel
```

Interop examples can declare extra cuda-oxide device crates with
`[[package.metadata.cuda-oxide.device-crates]]`, plus optional
`[package.metadata.cuda-oxide.interop]` metadata. `cargo oxide run` builds those
device crates with `rustc-codegen-cuda`, writes their PTX to the
configured location, and then builds/runs the host crate normally.
`cutile_inter_kernel` uses this path:
the host crate is a cutile-rs program, while `simt/` is a cuda-oxide SIMT PTX
crate loaded by the host at runtime.

### `cargo oxide sanitize <example>`

Builds an example like `cargo oxide run`, then executes the produced release
binary under NVIDIA Compute Sanitizer. `memcheck` is the default tool; use
`--tool racecheck`, `--tool initcheck`, or `--tool synccheck` for shared-memory
hazard, uninitialized global-memory, or synchronization checks.

```bash
cargo oxide sanitize vecadd --arch sm_75
cargo oxide sanitize sharedmem --tool racecheck
cargo oxide sanitize debug --tool synccheck -- --kernel-name kns=sync
cargo oxide sanitize vecadd -- --leak-check full
cargo oxide sanitize my_app -- --leak-check full -- --app-flag value
```

Arguments after the first `--` are passed to `compute-sanitizer` before the
executable. A second `--` splits target program arguments, matching
`compute-sanitizer [options] [your-program] [your-program-options]`.

`sanitize` uses Cargo's reported executable artifact, so custom binary names,
workspace layouts, configured target directories, and host target triples do
not require path guessing. Device line tables are enabled by default while
normal optimization remains on.

Compute Sanitizer's own error exit code defaults to zero. This command supplies
`--error-exitcode 86` unless you pass an explicit value, so tool findings fail
scripts and CI. If you intentionally pass `--error-exitcode 0`, inspect the
printed report; a zero process status no longer implies that the report was
clean. Options such as `--check-exit-code no` and `--require-cuda-init no` also
weaken what a zero status proves. The wrapper reports completion and reminds
you to inspect the sanitizer output instead of declaring the report clean from
status alone.
NVIDIA recommends treating the tools as complementary: `racecheck`,
`initcheck`, and `synccheck` do not replace `memcheck` memory-access checking.

### `cargo oxide build <example>`

Same as `run` but stops after compilation. Useful for examples that require hardware you don't have (e.g., Blackwell tensor cores).

```bash
cargo oxide build htens          # compiles PTX, doesn't try to run on GPU
cargo oxide build tcgen05        # sm_100a only, but PTX generation works anywhere
```

`build` also has a passthrough mode for normal Cargo workspaces. Put the Cargo
arguments after `--`; cargo-oxide supplies the backend, target architecture,
configured environment, and optional device owner filters.

```bash
cargo oxide build --                           # plain `cargo build`
cargo oxide build --arch sm_86 -- -p my_app --bin app --release
cargo oxide build --cargo-target-dir target/cuda -- -p my_app --release
cargo oxide build --device-codegen-crate gpu-kernels,math_gpu -- -p my_app
```

The owner filter is matched against rustc crate names, so Cargo package hyphens
are normalized to underscores (`gpu-kernels` matches `gpu_kernels`). The CUDA
backend still compiles host code for every crate through LLVM; it emits device
artifacts only for the listed owner crates.

The filter controls compilation, not runtime module names. Code in an excluded
crate or target must not call that target's generated `load()` function. In a
package with several targets, embedded bundles still use the package name, so
an excluded target's loader could otherwise find a selected sibling target's
bundle.

Passthrough builds include the effective codegen settings and backend build in
Cargo's rustc fingerprint. Changing the target, output mode, FMA policy, owner
list, configured codegen environment, or backend `.so` therefore regenerates
device artifacts instead of reusing a stale Cargo result.

### `cargo oxide emit-ltoir <crate>`

Compiles a crate's device code to a binary LTOIR artifact in one step, for the
Tile-to-SIMT interop workflow ([#96](https://github.com/NVlabs/cuda-oxide/issues/96)):
cuda-oxide is the SIMT participant, producing LTOIR that a tile or CUDA C++ kernel
links against. It builds the crate in NVVM IR mode, then runs the emitted
`<crate>.ll` through libNVVM `-gen-lto`, writing `<crate>.ltoir`.

`--arch` is required, since LTOIR is architecture-specific. It accepts `sm_XX`,
`compute_XX`, or a bare `XX`. The default output path is `<crate-dir>/<crate>.ltoir`;
`-o/--output` overrides it.

```bash
cargo oxide emit-ltoir standalone_device_fn --arch sm_100
cargo oxide emit-ltoir my_simt_crate --arch sm_120 -o build/simt.ltoir
```

cuda-oxide chooses the NVVM IR format from the selected architecture.
Pre-Blackwell targets use LLVM 7 typed pointers; `compute_100` and newer targets
use modern opaque pointers. The target is checked before export and recorded
alongside the artifact so the runtime uses the same format.

### `cargo oxide test`

Runs `cargo test` through the cuda-oxide backend. With no extra arguments it
runs the workspace's normal test selection. Put Cargo test arguments after
`--`, including the test-binary separator when needed.

```bash
cargo oxide test
cargo oxide test -- -p my_app --release --test gpu_smoke -- --nocapture
cargo oxide test --device-codegen-crate gpu_smoke -- -p my_app --test gpu_smoke
```

### `cargo oxide pipeline <example>`

Shows verbose progress plus the selected IR artifacts: MIR collection,
`dialect-mir` before and after `mem2reg`, LLVM dialect, LLVM IR, and final PTX.

```bash
cargo oxide pipeline vecadd
cargo oxide pipeline device_ffi_test --emit-nvvm-ir --arch sm_120
```

### `cargo oxide debug <example>`

Builds with debug info (`-C debuginfo=2`) and launches cuda-gdb. Supports
`--tui` for GDB's TUI mode and `--cgdb` for the cgdb frontend.

`debug` uses Cargo's reported executable artifact, so custom binary names,
`package.default-run`, configured target directories, host target triples, and
virtual workspaces do not require path guessing. Use `--bin <name>` when a
package has multiple binaries and no unambiguous default.

### `cargo oxide new <name> [--async]`

Scaffolds a new standalone cuda-oxide project with `Cargo.toml`, `rust-toolchain.toml`, and a working `src/main.rs` containing a vector addition kernel. The default template uses `#[cuda_module]` with typed synchronous launch methods; `--async` generates a template with `tokio`, `cuda-async`, and typed lazy `DeviceOperation` launches.

```bash
cargo oxide new my_kernel
cd my_kernel
cargo oxide run
```

### `cargo oxide fmt [--check]`

Formats all crates in the workspace: root workspace, `rustc-codegen-cuda`, and all examples. With `--check`, reports files that need formatting without modifying them.

### `cargo oxide doctor`

Validates that your environment is correctly set up: Rust nightly toolchain,
CUDA headers (`cuda.h`), CUDA toolkit (`nvcc`, libNVVM, nvJitLink,
libdevice), LLVM (`llc`), clang/libclang, the NVIDIA driver / GPU, and the
codegen backend `.so`. Every check reports what was found or how to fix it.

`cargo-oxide` itself builds and runs without the CUDA toolkit and without an
NVIDIA driver, and `doctor` never builds anything first, so it works on a
bare machine and tells you exactly what is missing. The driver / GPU check is
informational (only `cargo oxide run` needs a GPU), and a missing backend
`.so` just points at `cargo oxide setup` (`run`/`build` build it on demand
anyway).

### `cargo oxide setup`

Explicitly builds (or rebuilds) the codegen backend. Normally this happens automatically on every `run`/`build`/`pipeline` command, but `setup` is useful after pulling new changes or for CI.

## Backend Discovery

When `cargo oxide` needs the `librustc_codegen_cuda.so` backend, it searches in this order:

1. **`CUDA_OXIDE_BACKEND` env var** — explicit path override
2. **Project config** — `.cargo/cuda-oxide.toml`
3. **Local repo** — detects `crates/rustc-codegen-cuda` relative to workspace root, builds from source
4. **Cached `.so`** — checks `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`
5. **Auto-fetch** — clones the cuda-oxide repo, builds, and caches (one-time)

Project config can also provide the default architecture, extra rustc flags,
and child-process environment:

```toml
backend = "/path/to/librustc_codegen_cuda.so"
default-arch = "sm_86"
extra-rustflags = ["--cfg", "my_device_cfg"]

[env]
MY_BUILD_FLAG = "1"
```

Relative backend paths are resolved from the `.cargo` directory containing the
config file. Each `extra-rustflags` array element remains one rustc argument,
including values that contain spaces.

Configuration values are defaults. Precedence is:

1. explicit `cargo oxide` flags and internal artifact paths;
2. environment variables inherited by `cargo oxide`;
3. `.cargo/cuda-oxide.toml` defaults.

Do not put `RUSTFLAGS` or `CARGO_ENCODED_RUSTFLAGS` in the `[env]` table; use
`extra-rustflags` for project defaults. cargo-oxide combines configured flags,
inherited user flags, explicit `--device-cfg` values, and its required compiler
flags in boundary-preserving `CARGO_ENCODED_RUSTFLAGS`. Required compiler flags
are applied last so inherited settings cannot replace the cuda-oxide backend or
disable its correctness-critical codegen options.

## Architecture

```text
crates/cargo-oxide/
├── Cargo.toml
└── src/
    ├── main.rs       # CLI definitions (clap) + dispatch
    ├── backend.rs    # Backend discovery + build logic
    └── commands.rs   # All command implementations
```

## Future Commands

| Command                         | Description                                             |
|---------------------------------|---------------------------------------------------------|
| `cargo oxide bench <example>`   | GPU profiling (nsys/ncu integration), report TFLOPS     |
| `cargo oxide clean`             | Remove generated PTX/LL/LTOIR artifacts and build caches|
| `cargo oxide update`            | Update the cached codegen backend to latest version     |
| `cargo oxide list`              | List examples with descriptions and hardware reqs       |
| `cargo oxide inspect <example>` | Show generated PTX without the full pipeline dump       |
