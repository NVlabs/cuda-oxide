# cuda-oxide Windows Port

> **Fork of [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide)** with Windows support added.

## What is cuda-oxide?

cuda-oxide is an experimental compiler project by NVIDIA Labs that lets you write GPU kernels in **pure Rust** instead of CUDA C++. You annotate functions with `#[kernel]`, and a custom rustc codegen backend compiles them directly to PTX (NVIDIA GPU assembly).

```rust
#[cuda_module]
mod my_kernels {
    #[kernel]
    pub fn vector_add(a: &[f32], b: &[f32], mut out: DisjointSlice<f32>) {
        let tid = thread::index_1d();
        if let Some(slot) = out.get_mut(tid) {
            *slot = a[tid.get()] + b[tid.get()];
        }
    }
}
```

The compilation pipeline bypasses CUDA C++ entirely:

```
Rust source → rustc MIR → Pliron IR → LLVM IR → NVPTX → PTX
```

No `nvcc`. No `.cu` files. No C++ headers. Just Rust — with borrow checking, lifetimes, and zero-cost abstractions — compiled straight to GPU machine code.

## Why This Fork Exists

cuda-oxide was released as **Linux-only**. The README says so. The CI only runs on Linux. Every path in the codebase is hardcoded for ELF and `.so` shared libraries. There was no Windows support at all.

We needed it on Windows. So we ported it.

**This fork contains the minimum set of changes (6 fixes, ~60 lines of code) to make cuda-oxide compile and run on Windows.** Every existing example in the upstream repo works after these changes.

A [pull request (#227)](https://github.com/NVlabs/cuda-oxide/pull/227) has been submitted to upstream.

## What We Changed (and Why)

### Fix 1: CUDA Header Discovery (environment variable)

**Problem:** `cuda-bindings` uses `bindgen` to generate Rust FFI from `cuda.h`. It searches Linux paths like `/usr/local/cuda/include`. On Windows, CUDA installs elsewhere.

**Fix:** Set `CUDA_TOOLKIT_PATH` to your Windows CUDA install:
```powershell
$env:CUDA_TOOLKIT_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
```

### Fix 2: libclang for bindgen (environment variable)

**Problem:** `bindgen` requires `libclang` to parse C headers. On Windows you need `libclang.dll` from an LLVM installation.

**Fix:** Install LLVM and set:
```powershell
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"
```

### Fix 3: MSVC Enum Type Mismatch — `i32` vs `u32`

**Problem:** This is the most technically interesting fix. `bindgen` generates different types for C enums depending on the compiler:

| Platform | C Enum Type | Rust Type |
|----------|------------|-----------|
| Linux (GCC/Clang) | `unsigned int` | `u32` |
| Windows (MSVC) | `int` | `i32` |

MSVC defaults all C enums to signed `int`. GCC picks `unsigned int` when all values are positive. The entire `cuda-core` crate was written assuming `u32` because it was only tested on Linux.

This caused **10 type errors** across 4 files:

**Fix:** Added `as u32` casts at every call site. All CUDA enum constants are positive (bit flags like `CU_STREAM_NON_BLOCKING = 0x1`), so the cast is always safe.

**Files:** `cuda-core/src/context.rs`, `event.rs`, `lib.rs`, `stream.rs`

### Fix 4: PE/COFF 65535 Export Limit

**Problem:** The codegen backend (`rustc_codegen_cuda`) is built as a Rust `dylib` — a shared library that rustc loads at runtime. On Linux, the `.so` has no symbol export limit. On Windows, PE/COFF format limits DLL exports to **65,535 symbols**. The codegen backend re-exports ~66,953 symbols from `rustc_driver` (mostly LLVM internals). That's 1,418 over the limit.

**Fix:** Three new files:

1. **`codegen_backend.def`** — A minimal export definition that only exports `__rustc_codegen_backend`, the single entry point rustc needs:
   ```def
   EXPORTS
       __rustc_codegen_backend
   ```

2. **`build.rs`** — Tells the linker to use our `.def` file instead of the auto-generated one (which lists all 66,953 symbols):
   ```rust
   #[cfg(target_os = "windows")]
   {
       println!("cargo:rustc-link-arg=/DEF:{}", def_path.display());
   }
   ```

3. **`.cargo/config.toml`** — Uses LLVM's `lld-link` instead of MSVC's `link.exe`:
   ```toml
   [target.x86_64-pc-windows-msvc]
   linker = "C:\\Program Files\\LLVM\\bin\\lld-link.exe"
   ```

### Fix 5: PTX Embedding — ELF to COFF

**Problem:** After compiling `#[kernel]` functions to PTX, the bytecode gets embedded into the host executable as a data section inside an object file. The `oxide-artifacts` crate only knew how to create **ELF** object files (Linux). On Windows, the linker needs **COFF** object files.

**Fix:** Added Windows target detection and COFF section flags:

```rust
// Before: only ELF
section.flags = SectionFlags::Elf { sh_flags: SHF_ALLOC | SHF_GNU_RETAIN };

// After: platform-aware
match target.format {
    BinaryFormat::Elf  => { /* ELF flags */ }
    BinaryFormat::Coff => {
        section.flags = SectionFlags::Coff {
            characteristics: IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ,
        };
    }
    _ => {}
}
```

**File:** `oxide-artifacts/src/lib.rs`

### Fix 6: `.so` → `.dll` Path Detection

**Problem:** `cargo-oxide/src/backend.rs` had `librustc_codegen_cuda.so` hardcoded in 6 places. On Windows, the shared library is `rustc_codegen_cuda.dll` (no `lib` prefix, `.dll` extension).

**Fix:** Added a platform-aware helper:

```rust
fn backend_lib_name() -> &'static str {
    if cfg!(target_os = "windows") {
        "rustc_codegen_cuda.dll"
    } else {
        "librustc_codegen_cuda.so"
    }
}
```

**File:** `cargo-oxide/src/backend.rs`

## Quick Start (Windows)

### Prerequisites

- **Windows 10/11** (x86_64)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** 12.x or 13.x — [download](https://developer.nvidia.com/cuda-downloads)
- **LLVM** — `winget install LLVM.LLVM`
- **Rust nightly** — `rustup toolchain install nightly-2026-04-03`
- **Visual Studio Build Tools** — MSVC linker + Windows SDK

### Build

```powershell
# Clone this fork
git clone https://github.com/joshuapetersen/cuda-oxide.git
cd cuda-oxide

# Set environment
$env:CUDA_TOOLKIT_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"
$env:LIBCLANG_PATH = "C:\Program Files\LLVM\bin"

# Build the entire workspace (18 crates)
cargo +nightly-2026-04-03 build

# Build the codegen backend DLL
cd crates/rustc-codegen-cuda
cargo +nightly-2026-04-03 build
# Produces: target/debug/rustc_codegen_cuda.dll (23.8 MB)
cd ../..

# Build and run an example with GPU kernels
cargo +nightly-2026-04-03 oxide run vecadd
```

### Verify

If everything works, `cargo oxide run vecadd` will:
1. Load `rustc_codegen_cuda.dll` into the Rust compiler
2. Compile the `#[kernel]` function to PTX
3. Embed the PTX into a COFF object
4. Link the final executable against `cuda.lib`
5. Execute the kernel on your GPU

## Change Summary

```
 6 files changed, 57 insertions(+), 25 deletions(-)
 3 files created (build.rs, config.toml, codegen_backend.def)
```

| Crate | Files | What Changed |
|-------|-------|-------------|
| `cuda-core` | 4 | 10x `as u32` enum casts |
| `oxide-artifacts` | 1 | COFF object format + section flags |
| `cargo-oxide` | 1 | `.dll` path detection |
| `rustc-codegen-cuda` | 3 (new) | Export limit workaround |

## Tested Configuration

| Component | Version |
|-----------|---------|
| OS | Windows 11 |
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Compute Capability | SM_89 (Ada Lovelace) |
| CUDA Toolkit | 13.1 |
| Rust | nightly-2026-04-03 |
| LLVM | 22.1.7 |

## License

Same as upstream: Apache-2.0. See [LICENSE-APACHE](LICENSE-APACHE).
