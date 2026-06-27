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
cargo oxide pipeline vecadd         # verbose pipeline dump
                                    # (MIR -> dialect-mir -> LLVM dialect -> LLVM IR -> PTX)
cargo oxide debug vecadd --tui      # build + launch cuda-gdb
cargo oxide fmt                     # format all crates
cargo oxide fmt --check             # check formatting
cargo oxide doctor                  # validate environment
cargo oxide setup                   # explicitly build the codegen backend
```

### Flags

| Flag               | Applies to                       | Description                                     |
|--------------------|----------------------------------|-------------------------------------------------|
| `--emit-nvvm-ir`   | run, build, pipeline             | Generate NVVM IR for libNVVM                    |
| `--arch <sm_XX>`   | run, build, pipeline, emit-ltoir | Target arch override                            |
| `--features <F>`   | run, build, emit-ltoir           | Comma-separated cargo features to enable        |
| `-o, --output <P>` | emit-ltoir                       | Output path for the `.ltoir` artifact           |
| `-v, --verbose`    | run, build, emit-ltoir           | Show detailed compilation output                |
| `--no-fmad`        | run, build, pipeline             | Disable FMA contraction (keep separate mul+add) |
| `--async`          | new                              | Use the async template                          |
| `--cgdb`           | debug                            | Use cgdb instead of cuda-gdb                    |
| `--tui`            | debug                            | Use GDB's TUI interface                         |
| `--check`          | fmt                              | Check formatting only                           |

`--arch` is required for `emit-ltoir` and explicit NVVM IR output because those
artifacts are architecture-specific. Without an override, `run` detects the
local GPU, while `build` and `pipeline` use the compiler's feature-based target
so they remain useful for cross-compilation.

`--no-fmad` disables FMA contraction for kernels that rely on two separate
roundings (e.g. Dekker's algorithm, 2Sum). Equivalent to `CUDA_OXIDE_NO_FMA=1`.
By default, FMA contraction is on (matching nvcc's `--fmad=true`): `fmul+fadd`
pairs fuse into a single `fma.rn.f32` for fewer instructions and one rounding.

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

### `cargo oxide build <example>`

Same as `run` but stops after compilation. Useful for examples that require hardware you don't have (e.g., Blackwell tensor cores).

```bash
cargo oxide build htens          # compiles PTX, doesn't try to run on GPU
cargo oxide build tcgen05        # sm_100a only, but PTX generation works anywhere
```

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

### `cargo oxide pipeline <example>`

Shows verbose progress plus the selected IR artifacts: MIR collection,
`dialect-mir` before and after `mem2reg`, LLVM dialect, LLVM IR, and final PTX.

```bash
cargo oxide pipeline vecadd
cargo oxide pipeline device_ffi_test --emit-nvvm-ir --arch sm_120
```

### `cargo oxide debug <example>`

Builds with debug info (`-C debuginfo=2`) and launches cuda-gdb. Supports `--tui` for GDB's TUI mode and `--cgdb` for the cgdb frontend.

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
2. **Local repo** — detects `crates/rustc-codegen-cuda` relative to workspace root, builds from source
3. **Cached `.so`** — checks `~/.cargo/cuda-oxide/librustc_codegen_cuda.so`
4. **Auto-fetch** — clones the cuda-oxide repo, builds, and caches (one-time)

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
| `cargo oxide test`              | Run all examples as a test suite, report pass/fail      |
| `cargo oxide clean`             | Remove generated PTX/LL/LTOIR artifacts and build caches|
| `cargo oxide update`            | Update the cached codegen backend to latest version     |
| `cargo oxide list`              | List examples with descriptions and hardware reqs       |
| `cargo oxide inspect <example>` | Show generated PTX without the full pipeline dump       |
