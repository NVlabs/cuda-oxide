# Building from Source

This appendix walks through setting up a development environment for cuda-oxide
from a fresh checkout. If you just want to run an example, the
[Writing Your First Kernel](../getting-started/hello-gpu.md) chapter is faster.

---

## Requirements

| Dependency       | Version                       | Purpose                                                     |
|:-----------------|:----------------------------- |:------------------------------------------------------------|
| **Rust nightly** | `nightly-2026-04-03` (pinned) | Compiler toolchain with `rustc-dev` for the codegen backend |
| **CUDA Toolkit** | 12.x+                         | Driver API, `nvcc`, PTX assembler                           |
| **LLVM**         | 21+ with NVPTX backend        | `llc-21`/`llc-22` for LLVM IR → PTX lowering                |
| **Clang**        | 21+ (`clang-21` pkg)          | `bindgen` in host `cuda-bindings` needs clang's headers     |
| **Linux**        | Tested on Ubuntu 24.04        | Windows and macOS are not supported                         |
| **GPU**          | sm_80, sm_90, sm_100a         | Hardware target                                             |

## Clone the repository

```bash
git clone https://github.com/NVlabs/cuda-oxide.git
cd cuda-oxide
```

## Install the Rust toolchain

The repo ships a `rust-toolchain.toml` that pins the exact nightly version and
components. Rustup picks it up automatically:

```toml
# rust-toolchain.toml (already in the repo root)
[toolchain]
channel = "nightly-2026-04-03"
components = ["rust-src", "rustc-dev", "rust-analyzer"]
```

If you need to install manually:

```bash
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev --toolchain nightly-2026-04-03
```

`rust-src` provides the standard library source for cross-compilation and
`rustc-dev` exposes compiler internals that the codegen backend links against.

## Install CUDA

Make sure the CUDA toolkit is on your `PATH`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
nvcc --version   # should print 12.x or later
```

If you are building on a system without a GPU (e.g. CI), the toolkit is still
required for `ptxas` and header files, but you will not be able to run kernels.

## Install LLVM

The codegen pipeline emits LLVM IR and invokes `llc` to produce PTX. LLVM 21
or later is required — earlier `llc` releases reject the TMA / tcgen05 / WGMMA
intrinsic signatures that cuda-oxide emits. The NVPTX backend must be enabled:

```bash
# Ubuntu / Debian
sudo apt install llvm-21

For most Debian-based systems, you can install the official packages from the LLVM repository:
# Setup the official LLVM repository
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21

# Verify NVPTX support
llc-21 --version | grep nvptx
```

You should see `nvptx64 - NVIDIA PTX 64-bit` in the target list. The pipeline
auto-discovers `llc-22` and `llc-21` on `PATH` (in that order); to pin a
specific binary set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.

```{note}
Older `llc` binaries (LLVM 20 and earlier) will compile simpler kernels when
pointed at via `CUDA_OXIDE_LLC=/path/to/llc-20`, but any example that uses
modern TMA / tcgen05 / WGMMA intrinsics (`tma_copy`, `gemm_sol`,
`tcgen05_matmul`, `wgmma`, …) will fail with
`Intrinsic has incorrect argument type!` until you upgrade to LLVM 21+.
```

## Install Clang (for host `cuda-bindings`)

The host `cuda-bindings` crate runs `bindgen`, which loads libclang and needs
clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not
enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` verifies this up front.

## Build the workspace

The main workspace contains the user-facing crates (`cuda-device`, `cuda-core`,
`cuda-async`, etc.) and the build tooling (`cargo-oxide`):

```bash
cargo build
```

```{note}
The codegen backend (`crates/rustc-codegen-cuda/`) is intentionally **not** a
workspace member because it requires special nightly features and a different
build process. `cargo-oxide` handles building it transparently.
```

## Install cargo-oxide

`cargo-oxide` is the cargo subcommand that drives the full compilation
pipeline. Inside the repo, it works via a workspace alias. For standalone use:

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` automatically fetches and builds the codegen backend
dylib.

## Verify the installation

```bash
# Check all prerequisites
cargo oxide doctor

# Compile and run the canonical first example
cargo oxide run vecadd
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit (including
libNVVM / nvJitLink / libdevice for kernels that use math intrinsics), LLVM
installation, and codegen backend. If everything is configured correctly,
`cargo oxide run vecadd` compiles a Rust kernel to PTX, launches it on the GPU,
and prints a success message.

## Common commands

```bash
# Build and run an example
cargo oxide run <example>

# Show the full compilation pipeline (MIR → LLVM IR → PTX)
cargo oxide pipeline <example>

# Debug with cuda-gdb
cargo oxide debug <example> --tui

# Build with LTOIR output (for C++ interop)
cargo oxide build <example> --dlto
```

## Building the book

The documentation lives in `cuda-oxide-book/` and uses Sphinx with MyST
Markdown. To build and serve locally:

```bash
cd cuda-oxide-book
make setup      # creates venv, installs dependencies
source .venv/bin/activate
make livehtml   # starts dev server on http://localhost:8000
```

## Generating API documentation

Standard `cargo doc` works for the workspace crates:

```bash
cargo doc --no-deps --open
```

This generates rustdoc for `cuda-device`, `cuda-core`, `cuda-async`, and all
other workspace members. The codegen backend is excluded since it is not a
workspace member.

## Workspace structure

```text
cuda-oxide/
├── Cargo.toml              # Workspace root
├── rust-toolchain.toml     # Pinned nightly + components
├── crates/
│   ├── cuda-device/          # Device intrinsics (#![no_std])
│   ├── cuda-host/            # cuda_launch! macro
│   ├── cuda-macros/          # Proc macros (#[kernel], #[device], gpu_printf!)
│   ├── cuda-bindings/        # Raw bindgen FFI to cuda.h
│   ├── cuda-core/            # Safe RAII wrappers (CudaContext, DeviceBuffer)
│   ├── cuda-async/           # Async execution (DeviceOperation, DeviceFuture)
│   ├── cargo-oxide/          # Cargo subcommand
│   ├── rustc-codegen-cuda/   # Codegen backend (not a workspace member)
│   ├── mir-importer/         # MIR → Pliron IR translation
│   ├── mir-lower/            # `dialect-mir` → `dialect-llvm` lowering
│   ├── dialect-mir/          # pliron dialect modelling Rust MIR
│   ├── dialect-llvm/         # pliron dialect modelling LLVM IR (+ export)
│   ├── dialect-nvvm/         # NVVM intrinsics dialect
│   ├── libnvvm-sys/          # dlopen bindings to libNVVM
│   ├── nvjitlink-sys/        # dlopen bindings to nvJitLink
│   ├── reserved-oxide-symbols/ # Shared naming contract
│   └── fuzzer/               # Differential testing support
└── cuda-oxide-book/        # This book (Sphinx + MyST)
```

## Troubleshooting

`llc` not found or missing NVPTX
: Ensure LLVM 21+ is installed and `llc-21` (or `llc-22`) is on your `PATH`.
  The pipeline probes `llc-22` → `llc-21` in order; alternatively set
  `CUDA_OXIDE_LLC=/usr/bin/llc-21` to point at a specific binary.

`Intrinsic has incorrect argument type!` (from `llc`)
: Your `llc` is older than LLVM 21 and cannot lower the modern TMA / tcgen05
  / WGMMA intrinsic signatures. Install `llvm-21` and re-run.

`error[E0463]: can't find crate for rustc_middle`
: You are missing the `rustc-dev` component. Run:
  `rustup component add rustc-dev --toolchain nightly-2026-04-03`.

CUDA driver version mismatch
: The toolkit version (compile-time) and driver version (runtime) must be
  compatible. Run `nvidia-smi` to check the driver version, and
  `nvcc --version` for the toolkit.

`cargo oxide doctor` fails on codegen backend
: The backend is built on first use. If the build fails, check that
  `rust-src` is installed and that the nightly version matches
  `rust-toolchain.toml`.
