# cargo-oxide

Cargo subcommand for building and running Rust GPU programs with cuda-oxide.

Replaces the previous `xtask` pattern with a proper cargo subcommand that works both inside the cuda-oxide repo (for developers) and externally (for users who `cargo install`).

## Installation

**Internal developers** (inside the cuda-oxide repo): no installation needed. The workspace alias makes `cargo oxide` work immediately.

**External users**:

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend if it's not already available.

## Usage

```bash
cargo oxide new my_project          # scaffold a new cuda-oxide project
cargo oxide new my_project --async  # scaffold with async template (tokio + cuda-async)
cargo oxide run vecadd              # build + run an example
cargo oxide build vecadd            # compile only (no run)
cargo oxide pipeline vecadd         # verbose pipeline dump
                                    # (MIR -> dialect-mir -> dialect-llvm -> LLVM IR -> PTX)
cargo oxide debug vecadd --tui      # build + launch cuda-gdb
cargo oxide fmt                     # format all crates
cargo oxide fmt --check             # check formatting
cargo oxide doctor                  # validate environment
cargo oxide setup                   # explicitly build the codegen backend
```

### Flags

| Flag              | Applies to           | Description                              |
|-------------------|----------------------|------------------------------------------|
| `--emit-nvvm-ir`  | run, build, pipeline | Generate NVVM IR for libNVVM             |
| `--arch <sm_XX>`  | run, build, pipeline | Target architecture override             |
| `--features <F>`  | run, build           | Comma-separated cargo features to enable |
| `-v, --verbose`   | run, build           | Show detailed compilation output         |
| `--async`         | new                  | Use the async template                   |
| `--cgdb`          | debug                | Use cgdb instead of cuda-gdb             |
| `--tui`           | debug                | Use GDB's TUI interface                  |
| `--check`         | fmt                  | Check formatting only                    |

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
```

### `cargo oxide build <example>`

Same as `run` but stops after compilation. Useful for examples that require hardware you don't have (e.g., Blackwell tensor cores).

```bash
cargo oxide build htens          # compiles PTX, doesn't try to run on GPU
cargo oxide build tcgen05        # sm_100a only, but PTX generation works anywhere
```

### `cargo oxide pipeline <example>`

Shows the full compilation pipeline with verbose output at every stage: MIR collection, `dialect-mir` (alloca + post-`mem2reg`), `dialect-llvm`, textual LLVM IR, and the final PTX.

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

Validates that your environment is correctly set up: Rust nightly toolchain, CUDA toolkit (`nvcc`), LLVM (`llc`), and the codegen backend `.so`.

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
