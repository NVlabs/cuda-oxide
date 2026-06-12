<p align="center">
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/clippy.yml"><img alt="clippy" src="https://github.com/NVlabs/cuda-oxide/actions/workflows/clippy.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/unit-tests.yml"><img alt="unit-tests" src="https://github.com/NVlabs/cuda-oxide/actions/workflows/unit-tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/cargo-deny.yml"><img alt="cargo-deny" src="https://github.com/NVlabs/cuda-oxide/actions/workflows/cargo-deny.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/NVlabs/cuda-oxide/actions/workflows/codeql.yml"><img alt="CodeQL" src="https://github.com/NVlabs/cuda-oxide/actions/workflows/codeql.yml/badge.svg?branch=main"></a>
  <br>
  <img src="assets/logo.png" alt="cuda-oxide logo" width="100%">
</p>

# cuda-oxide

cuda-oxide is a custom rustc backend for compiling GPU kernels in pure Rust.
The workspace combines:

- single-source compilation -- host and device code live in the same file, built with one `cargo oxide build`
- a rustc codegen backend that compiles `#[kernel]` functions to CUDA PTX
- device-side abstractions (type-safe indexing, shared memory, scoped atomics, barriers, TMA, warp/cluster ops)
- a host-side runtime for memory management, pinned host transfers, and kernel launching (`cuda-core`, `cuda-async`)
- a rust-native compilation pipeline using [Pliron](https://github.com/vaivaswatha/pliron), an MLIR-like IR framework in Rust (Rust → Rust MIR → Pliron IR → LLVM IR → PTX)

## Project Status

cuda-oxide is an experimental compiler that demonstrates how CUDA SIMT kernels can be written natively in pure Rust -- no DSLs, no foreign language bindings -- and made available to the broader Rust community. The project is in an early stage (alpha) and under active development: you should expect bugs, incomplete features, and API breakage as we work to improve it. That said, we hope you'll try it in your own work and help shape its direction by sharing feedback on your experience.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) if you're interested in contributing to the project.

## Quick Start

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

// Device: generic kernel that applies any function to each element.
// F can be a closure with captures — rustc monomorphizes it to a concrete type.
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn map<T: Copy, F: Fn(T) -> T + Copy>(f: F, input: &[T], mut out: DisjointSlice<T>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(out_elem) = out.get_mut(idx) {
            *out_elem = f(input[i]);
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let input = DeviceBuffer::from_host(&stream, &data).unwrap();
    let mut output = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    let module = kernels::load(&ctx).unwrap();

    // Launch with a closure — factor is captured and passed to the GPU automatically
    let factor = 2.5f32;
    module
        .map::<f32, _>(
            &stream,
            LaunchConfig::for_num_elems(1024),
            move |x: f32| x * factor,
            &input,
            &mut output,
        )
        .unwrap();

    let result = output.to_host_vec(&stream).unwrap();
    assert!((result[1] - 2.5).abs() < 1e-5);
}
```

The above example defines a generic `#[kernel]` function `map` that accepts any
`Fn(T) -> T` closure. `#[cuda_module]` embeds the generated device artifact into
the host binary and generates a typed `module.map::<f32, _>(...)` launch method.
The closure `move |x| x * factor` is captured, scalarized, and passed as kernel
parameters automatically.

For composable async GPU work, `stream:` disappears, `{kernel}_async` returns a
lazy `DeviceOperation`, and execution happens when you call `.sync()` or
`.await`.

```rust
use cuda_async::device_operation::DeviceOperation;

// Assuming `module`, `input`, and `output` come from the cuda-async setup:
let factor = 2.5f32;
module
    .map_async::<f32, _>(
        LaunchConfig::for_num_elems(1024),
        move |x: f32| x * factor,
        &input,
        &mut output,
    )?
    .sync()?;
// or: .await?;
```

See the `async_mlp` example and `crates/cuda-async/README.md` for the full async setup.

```bash
# Build and run an example
cargo oxide run host_closure

# Show full compilation pipeline (Rust MIR → dialect-mir → mem2reg → LLVM dialect → LLVM IR → PTX)
cargo oxide pipeline vecadd

# Debug with cuda-gdb
cargo oxide debug vecadd --tui
```

## Setup

### Requirements

- **cargo-oxide** — cargo subcommand that drives the build pipeline (`cargo oxide run`, `build`, `debug`, etc.)
- **Rust nightly** with `rust-src` and `rustc-dev` and `llvm-tools` components (pinned in `rust-toolchain.toml`)
- **CUDA Toolkit** (12.x+)
- **Clang + libclang dev headers** (`clang-21` / `libclang-common-21-dev`) — needed by `bindgen` when building the host `cuda-bindings` crate
- **Linux** (tested on Ubuntu 24.04)

### Install

#### cargo-oxide

Inside the cuda-oxide repo, `cargo oxide` works out of the box via a workspace alias.

For use outside the repo (your own projects), install it with the pinned nightly toolchain:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend.

#### Nix (alternative)

If you have Nix with flakes enabled, `nix develop` in the repo gives you a reproducible shell with CUDA 13, LLVM 22, Clang, and the pinned Rust nightly — no manual apt installs. The shellHook auto-discovers host NVIDIA drivers on NixOS and non-NixOS systems.

```bash
nix develop                                       # full dev shell in this repo
nix run github:NVlabs/cuda-oxide#new my-project   # bootstrap a project
```

#### Rust

```bash
# Toolchain installed automatically via rust-toolchain.toml
# Manual install if needed:
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03
```

#### CUDA

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version
```

#### LLVM (optional)

```bash
# Ubuntu/Debian
sudo apt install llvm-21
```

If your distro packages do not provide `llvm-21`, use LLVM's apt helper:

```bash
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

```bash
# Verify NVPTX support
llc-21 --version | grep nvptx
```

The pipeline prefers `llc` in Rust toolchain, and auto-discovers `llc-22` and `llc-21` on `PATH` (in that order).
To pin a specific binary, set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.

> We emit TMA / tcgen05 / WGMMA intrinsics that `llc` from LLVM 20 and earlier can't handle.
> Simple kernels might still work with an older `llc`, but anything Hopper / Blackwell needs 21+.

#### Clang (host `cuda-bindings`)

The host `cuda-bindings` crate runs `bindgen`, which loads libclang and needs
clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not
enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` catches this up front; the symptom otherwise is a cryptic
`'stddef.h' file not found` during the host build.

#### Dev Container

The repository includes a standard devcontainer setup in `.devcontainer/` for a
reproducible CUDA, LLVM, Clang, and Rust environment. See the
[installation chapter](cuda-oxide-book/getting-started/installation.md#dev-container)
for editor and CLI usage.

### Verifying Installation

```bash
# Check that all prerequisites are in place
cargo oxide doctor

# Build and run an example end-to-end
cargo oxide run vecadd
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit, LLVM, and
codegen backend. If everything is configured correctly, `cargo oxide run vecadd`
compiles a Rust kernel to PTX, launches it on the GPU, and prints
`✓ SUCCESS: All 1024 elements correct!`.

## Examples

**60+ examples** in `crates/rustc-codegen-cuda/examples/`. Highlights:

| Example              | Description                                                              |
|----------------------|--------------------------------------------------------------------------|
| `vecadd`             | Vector addition -- canonical first example                               |
| `host_closure`       | Generic kernels with closures passed from host                           |
| `generic`            | Generic kernels with monomorphization (`scale<T>`)                       |
| `ord_cmp`            | Device-side `Ord::cmp` lowering for signed and unsigned integers         |
| `gemm_sol`           | GEMM SoL: 868 TFLOPS, 58% cublasLt SoL on B200 (148 SMs); 8 kernels      |
| `tcgen05`            | Blackwell tensor cores (sm_100a): TMEM, MMA, cta_group::2                |
| `atomics`            | GPU atomics: 6 types x 3 scopes x 5 orderings (20 tests)                 |
| `cluster`            | Thread Block Clusters + DSMEM ring exchange (Hopper+)                    |
| `async_mlp`          | Async MLP pipeline: GEMM → MatVec → ReLU across concurrent streams       |
| `mathdx_ffi_test`    | cuFFTDx thread-level FFT + cuBLASDx block-level GEMM                     |
| `device_ffi_test`    | Device FFI: Rust kernels calling C++ CCCL warp-level reductions via LTOIR|
| `async_vecadd`       | Async GPU execution with `cuda-async` and `DeviceOperation`              |
| `cross_crate_kernel` | Library crates defining kernels, bundled into binaries                   |
| `cuda_module_in_lib` | `#[cuda_module]` in a library crate, loaded by embedded bundle name      |

```bash
cargo oxide run vecadd
cargo oxide run gemm_sol
```

## Crate Overview

### User-Facing Crates

| Crate               | Description                                                               |
|---------------------|---------------------------------------------------------------------------|
| `cuda-device`       | Device intrinsics (`thread::*`, `warp::*`, barriers)                      |
| `cuda-host`         | Typed module loading, launch helpers, LTOIR loader                        |
| `cuda-macros`       | Proc macros (`#[cuda_module]`, `#[kernel]`, `gpu_printf!`)                |
| `cuda-bindings`     | Raw `bindgen` FFI bindings to `cuda.h`                                    |
| `cuda-core`         | Safe RAII wrappers (`CudaContext`, `CudaStream`, `DeviceBuffer<T>`, ...)  |
| `cuda-async`        | Async execution layer (`DeviceOperation`, `DeviceFuture`, `DeviceBox<T>`) |
| `libnvvm-sys`       | `dlopen` bindings to libNVVM (used by `cuda-host::ltoir`)                 |
| `nvjitlink-sys`     | `dlopen` bindings to nvJitLink (used by `cuda-host::ltoir`)               |

### Compiler Crates

| Crate                | Description                                           |
|----------------------|-------------------------------------------------------|
| `rustc-codegen-cuda` | Custom rustc backend                                  |
| `mir-importer`       | Rust MIR -> `dialect-mir` translation + pipeline      |
| `mir-lower`          | `dialect-mir` -> LLVM dialect lowering                |
| `dialect-mir`        | pliron dialect modelling Rust MIR                     |
| `llvm-export`        | pliron-llvm shim + textual `.ll` exporter             |
| `dialect-nvvm`       | pliron dialect modelling NVVM intrinsics              |

### Build Tooling

| Crate          | Description                                          |
|----------------|------------------------------------------------------|
| `cargo-oxide`  | Cargo subcommand (`cargo oxide run`, etc.)           |

### Documentation

| Directory           | Description                                                        |
|---------------------|--------------------------------------------------------------------|
| `cuda-oxide-book`   | Project book (Sphinx + MyST) — guides, compiler internals, API ref |

## Status

### Highlights:

- End-to-end Rust -> PTX compilation
- Unified single-source compilation (host + device in one file)
- Generic functions with monomorphization
- Closures with captures (move and non-move via HMM)
- User-defined structs, enums, pattern matching
- Full GPU intrinsic support (thread, warp, shared memory, barriers, TMA, clusters, atomics)
- Cross-crate kernels
- LTOIR generation for Blackwell+ (device-side LTO)
- Device FFI: Rust <-> C++/CCCL interop via LTOIR
- MathDx integration: cuFFTDx thread-level FFT, cuBLASDx block-level GEMM
- Tile interop (experimental): [`cutile_inter_kernel`](crates/rustc-codegen-cuda/examples/cutile_inter_kernel/README.md) chains a cutile-rs Tile kernel and a cuda-oxide SIMT PTX kernel on the same CUDA stream over shared device tensors. Intra-kernel Tile interop is work in progress and tracked in [#96](https://github.com/NVlabs/cuda-oxide/issues/96).
- Host runtime: `cuda-core` (explicit control, pinned host transfers) and `cuda-async` (composable async operations)
- GEMM SoL: 868 TFLOPS (58% of cublasLt FP16 SoL) on B200 (148 SMs) with cta_group::2 + CLC + 4-stage pipeline (`gemm_sol` example measures the cublasLt baseline live via `bench/cublaslt_bench` — absolute TFLOPS scale with SM count on smaller Blackwell DC SKUs; per-phase tables for both 148-SM and 80-SM variants are in `crates/rustc-codegen-cuda/examples/gemm_sol/README.md`)

## Documentation

**WIP:** 🚧 The **[cuda-oxide book](https://nvlabs.github.io/cuda-oxide/)** is the primary reference for the project. It covers SIMT kernel authoring in Rust, synchronous and asynchronous GPU programming, the compiler architecture, and more.

To build and serve the book locally, see [cuda-oxide-book/README.md](./cuda-oxide-book/README.md).

## Ecosystem

cuda-oxide is one of several Rust + GPU efforts under active development. Projects in this space address different parts of the problem — Vulkan/SPIR-V for graphics, implicit offload via LLVM, third-party CUDA backends, safe driver bindings — and we've been working with maintainers across the broader Rust GPU community on how to move GPU computing in Rust forward together. For where cuda-oxide fits relative to other projects, see the [Ecosystem appendix](https://nvlabs.github.io/cuda-oxide/appendix/ecosystem.html) of the book.

## License

The `cuda-bindings` crate is licensed under the NVIDIA Software License: [LICENSE-NVIDIA](LICENSE-NVIDIA). All other crates are licensed under the Apache License, Version 2.0: [LICENSE-APACHE](LICENSE-APACHE).
