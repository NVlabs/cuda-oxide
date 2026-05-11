# Installation

This section walks through everything you need to get `cargo oxide run vecadd` working on your machine.

---

## Prerequisites

| Requirement      | Version             | Notes                                                         |
|------------------|---------------------|---------------------------------------------------------------|
| **Linux**        | Ubuntu 24.04 tested | Other distros may work but are untested                       |
| **NVIDIA GPU**   | Ampere+ (sm_80+)    | Driver 545+ recommended                                       |
| **CUDA Toolkit** | 12.x+               | `nvcc` and `cuda.h` must be available                         |
| **LLVM**         | 21+                 | Must include the NVPTX backend                                |
| **Clang**        | 21+                 | `clang-21` — needed by `bindgen` for host `cuda-bindings`     |
| **Rust**         | Nightly (pinned)    | Pinned in `rust-toolchain.toml`                               |

:::{note}
cuda-oxide currently targets **Linux only**. Windows is not supported.
:::

---

## CUDA Toolkit

Install the CUDA Toolkit from the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page, then make sure it is on your `PATH`:

```bash
export PATH="/usr/local/cuda/bin:$PATH"
```

Verify the install:

```bash
nvcc --version
```

:::{tip}
If you installed CUDA to a non-default location, set `CUDA_TOOLKIT_PATH` to its root directory (the one containing `include/cuda.h`). If unset, cuda-oxide defaults to `/usr/local/cuda`.
:::

---

## LLVM (21+)

cuda-oxide uses LLVM's NVPTX backend to lower LLVM IR to PTX. Install LLVM 21 or newer and make sure `llc-21` (or `llc-22`) is on your `PATH`:

```bash
# Ubuntu / Debian
sudo apt install llvm-21

For most Debian-based systems, you can install the official packages from the LLVM repository:
# Setup the official LLVM repository
sudo apt-get install -y lsb-release wget software-properties-common gnupg
wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh
sudo ./llvm.sh 21
```

Verify that the NVPTX target is present:

```bash
llc-21 --version | grep nvptx
```

You should see a line containing `nvptx64` in the registered targets. The
pipeline auto-discovers `llc-22` and `llc-21` in that order; pin a specific
binary with `CUDA_OXIDE_LLC=/usr/bin/llc-21` if needed.

:::{warning}
A stock LLVM build without the NVPTX backend will not work. The `llvm-21` Ubuntu package includes it by default, but if you build LLVM from source you must pass `-DLLVM_TARGETS_TO_BUILD="X86;NVPTX"` to CMake.
:::

:::{important}
**Why LLVM 21?** We emit TMA / tcgen05 / WGMMA intrinsics that `llc` from LLVM 20 and earlier can't handle. Simple kernels might still work with an older `llc` (set `CUDA_OXIDE_LLC=/path/to/llc-20`), but anything Hopper / Blackwell needs 21+.
:::

---

## Clang (host `cuda-bindings`)

The host `cuda-bindings` crate runs [`bindgen`](https://github.com/rust-lang/rust-bindgen), which loads libclang and needs clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not enough.

```bash
sudo apt install clang-21   # or libclang-common-21-dev
```

`cargo oxide doctor` catches this up front; the symptom otherwise is `'stddef.h' file not found` during the host build.

---

## Rust toolchain

The workspace ships a `rust-toolchain.toml` that pins the exact nightly version and required components. When you first run any `cargo` command inside the repo, `rustup` will install the correct toolchain automatically.

If you need to install it manually:

```bash
rustup toolchain install nightly-2026-04-03
rustup component add rust-src rustc-dev rust-analyzer --toolchain nightly-2026-04-03
```

The two extra components are required by the codegen backend:

- `rust-src` -- source of the Rust standard library, needed for cross-compiling to the NVPTX target.
- `rustc-dev` -- compiler internals that the backend links against.

---

## cargo-oxide

`cargo-oxide` is the cargo subcommand that drives the entire build pipeline (`cargo oxide run`, `build`, `debug`, `pipeline`, etc.).

**Inside the cuda-oxide repo**, it works out of the box via a workspace alias -- no extra install step.

**For use outside the repo** (your own projects):

```bash
cargo install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend. Subsequent runs reuse the cached build.

---

## Verifying your installation

Run the built-in diagnostics check:

```bash
cargo oxide doctor
```

`cargo oxide doctor` validates your Rust toolchain, CUDA toolkit (including libNVVM / nvJitLink / libdevice for kernels that use math intrinsics), LLVM installation, and codegen backend in one shot.

Then build and run an example end-to-end:

```bash
cargo oxide run vecadd
```

If everything is configured correctly, this compiles a Rust kernel to PTX, launches it on the GPU, and prints a success message.

:::{tip}
**Common issues:**

- `No working llc-21 or llc-22 found on PATH` -- install LLVM 21+ (`sudo apt install llvm-21`), add `/usr/lib/llvm-21/bin` to your `PATH`, or set `CUDA_OXIDE_LLC=/usr/bin/llc-21`.
- `'stddef.h' file not found` when building host `cuda-bindings` -- install clang dev headers: `sudo apt install clang-21` (or `libclang-common-21-dev`).
- `cuda.h not found` -- Set `CUDA_TOOLKIT_PATH` to your CUDA install root, or ensure `/usr/local/cuda/include/cuda.h` exists.
- `rust-src component missing` -- Run `rustup component add rust-src --toolchain nightly-2026-04-03`.
:::
