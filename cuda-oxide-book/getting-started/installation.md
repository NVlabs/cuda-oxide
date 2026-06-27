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

## Dev Container

The repository includes a standard devcontainer setup in `.devcontainer/`.
Using it is the quickest way to get a reproducible development environment with
CUDA Toolkit 13.0, LLVM 21, Clang 21, and the pinned Rust nightly already
installed.

The host does not need the CUDA Toolkit installed. It does need:

- an NVIDIA GPU
- an NVIDIA driver compatible with CUDA 13.0
- Docker with the NVIDIA Container Toolkit installed

With a devcontainer-aware editor, open the repository and choose "Reopen in
Container" when prompted. The editor reads `.devcontainer/devcontainer.json`,
builds the image, requests GPU access with `--gpus=all`, and opens the checkout
inside the container.

For CLI-only usage, start the container with:

```bash
npx -y @devcontainers/cli up --workspace-folder .
```

Then run commands inside it with:

```bash
npx -y @devcontainers/cli exec --workspace-folder . cargo oxide doctor
npx -y @devcontainers/cli exec --workspace-folder . cargo oxide run vecadd
```

If the host driver is too old, GPU commands such as `nvidia-smi`,
`cargo oxide doctor`, or `cargo oxide run vecadd` will fail inside the
container. Update the host NVIDIA driver rather than installing a different
CUDA Toolkit in the container.

If you use the devcontainer, you can skip the manual CUDA, LLVM, Clang, and
Rust setup sections below.

---

## Nix / flake.nix

The repository also ships a `flake.nix` providing a reproducible dev shell
(CUDA 13, LLVM 22, Clang, pinned Rust nightly). Requires
[Nix](https://nixos.org/download/) with flakes enabled, an NVIDIA driver on
the host, and Linux (x86\_64 or aarch64).

Inside the cuda-oxide repo — `cargo-oxide` is included in the shell:

```bash
nix develop
cargo oxide run vecadd
```

To bootstrap a new project without cloning:

```bash
nix run github:NVlabs/cuda-oxide#new my-project
cd my-project && nix develop
```

This scaffolds via `cargo oxide new` and drops in a `flake.nix` that inherits
this repo's dev shell. The shellHook auto-discovers host NVIDIA driver
libraries on NixOS and non-NixOS systems; if the host driver is too old,
update it rather than changing what's inside the Nix shell.

If you use the Nix flake, you can skip the manual CUDA, LLVM, Clang, and
Rust setup sections below.

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

(installation-toolkit-driver-compatibility)=

### Toolkit and driver compatibility

The CUDA toolkit and NVIDIA driver are separate installations. A newer toolkit
can often run with an older driver through CUDA minor-version compatibility,
but that compatibility does not cover newer PTX:

```text
cubin ───────────────────────────► driver loads finished GPU code
PTX   ──► driver compiles it ─────► finished GPU code
```

The normal cuda-oxide path gets PTX from LLVM `llc`, so installing a newer CUDA
toolkit does not by itself change that PTX version. The NVVM forward-execution
path gets PTX from the selected toolkit's nvJitLink. If that PTX is newer than
the driver understands, module loading returns
`CUDA_ERROR_UNSUPPORTED_PTX_VERSION` (error 222).

For example, the toolchains used during current validation produced:

```text
LLVM llc ───────────────► PTX 8.7 ──► driver 580 ✓
CUDA 13.3 nvJitLink ───► PTX 9.3 ──► driver 580 ✗ error 222
```

These are observed versions, not permanent properties of the two paths. Check
the generated PTX `.version` directive when diagnosing another toolchain.

CUDA 13.x minor-version compatibility starts at driver 580, while CUDA 13.3's
corresponding full-support driver is 610.43.02. A 580 driver can therefore load
compatible finished cubins, but it cannot JIT PTX 9.3 produced by CUDA 13.3.
See the [CUDA release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
for the current driver table.

Upgrade the driver or select a compatible toolkit for the command:

```bash
CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0 cargo oxide run <example> --arch sm_86
```

See NVIDIA's
[minor-version compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
for the supported driver ranges and PTX limitation.

---

## LLVM 21+ (optional)

cuda-oxide uses LLVM's NVPTX backend to lower LLVM IR to PTX.

Usually `llc` in Rust toolchain is enough.

Install LLVM 21 or newer and make sure `llc-21` (or `llc-22`) is on your `PATH`:

```bash
# Ubuntu / Debian
sudo apt install llvm-21
```

If your distro packages do not provide `llvm-21`, use LLVM's apt helper:

```bash
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

The host `cuda-bindings` crate runs [`bindgen`](https://github.com/rust-lang/rust-bindgen), which loads `libclang.so` at runtime and needs clang's own resource-dir `stddef.h` — a bare `libclang1-*` runtime is not enough. Three packages cover both halves:

```bash
sudo apt install libclang-21-dev libclang-cpp21-dev libclang-common-21-dev
```

`cargo oxide doctor` catches this up front; the symptom otherwise is `'stddef.h' file not found` during the host build.

:::{tip}
**If doctor still reports clang not found:**
The issue may be that the versioned binary (e.g., clang-21) is installed but not mapped to the unversioned clang command required by bindgen. You can fix this by installing the clang meta-package, which manages these dependencies automatically:

```bash
sudo apt install clang libclang-dev
```

:::

:::{note}
**Fresh Ubuntu 24.04 / DGX-OS:** after installing LLVM 21 via `apt.llvm.org/llvm.sh` as shown above, the versioned `clang-21` / `clang++-21` binaries are present but the unversioned aliases `cargo oxide doctor` looks for are not. Add them with `update-alternatives`:

```bash
sudo update-alternatives --install /usr/bin/clang   clang   /usr/bin/clang-21   100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-21 100
```

Validated on Asus GX10 / NVIDIA DGX Spark.
:::

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

**For use outside the repo** (your own projects), install it with the pinned nightly toolchain:

```bash
cargo +nightly-2026-04-03 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide
```

On first run, `cargo-oxide` will automatically fetch and build the codegen backend. Subsequent runs reuse the cached build.

---

## Verifying your installation

Run the built-in diagnostics check:

```bash
cargo oxide doctor
```

`cargo oxide doctor` validates your Rust toolchain, CUDA headers (`cuda.h`),
CUDA toolkit (including libNVVM / nvJitLink / libdevice for kernels that use
math intrinsics), LLVM installation, NVIDIA driver / GPU, and codegen backend
in one shot.

`cargo-oxide` itself does not need the CUDA toolkit or an NVIDIA driver to
build and run, so `doctor` works even on a machine where nothing is installed
yet and diagnoses exactly what is missing (for example missing `cuda.h`, or
no driver). The driver / GPU check is informational: only `cargo oxide run`
needs a GPU, while `build` and `pipeline` work without one.

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
