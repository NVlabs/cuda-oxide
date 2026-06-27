# rustc-codegen-cuda

A custom rustc codegen backend that enables single-source CUDA programming in Rust. It intercepts rustc's code generation phase to split device code from host code -- device functions compile to PTX via the cuda-oxide pipeline, while host code passes through to the standard LLVM backend.

## Why Single-Source?

The alternative is split compilation: kernels live in a separate crate compiled with `#[cfg(cuda_device)]`, requiring two compilation passes, careful type coordination, and `instantiate!` macros for generics. With `rustc-codegen-cuda`, everything compiles in **one pass** -- host code and `#[kernel]` functions coexist in the same crate, share types naturally, and generics just work.

```rust
use cuda_device::{kernel, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}

fn main() {
    // Host code references the kernel directly.
    // PTX is generated alongside the host binary.
}
```

## How It Works

```text
  Source (.rs)
      │
      ▼
  rustc frontend  (parse → HIR → MIR → MIR optimizations)
      │
      ▼
  rustc_codegen_cuda
      ├── collect #[kernel] functions + transitive callees
      ├── device MIR  → dialect-mir → mem2reg → unroll → LLVM dialect → LLVM IR → PTX
      └── host   MIR  → rustc_codegen_llvm (standard path)
      │
      ▼
  host binary + PTX file(s)
```

The backend implements rustc's `CodegenBackend` trait. When rustc calls `codegen_crate()`:

1. **Collect** -- `collector.rs` scans codegen units for functions prefixed with `cuda_oxide_kernel_<hash>_` (set by the `#[kernel]` proc-macro -- the prefix is owned by `crates/reserved-oxide-symbols/`, the workspace-internal source of truth for the cuda-oxide naming contract). It then walks the MIR call graph to gather all transitively reachable device functions.
2. **Compile device code** -- `device_codegen.rs` feeds the collected MIR through
   the cuda-oxide pipeline: translate to `dialect-mir`, run `mem2reg`, apply
   annotated unrolling, lower and export to LLVM IR, then compile PTX with `llc`.
3. **Compile host code** -- The standard `rustc_codegen_llvm` backend handles everything else.

## Usage

```bash
# Preferred: use the cargo-oxide wrapper
cargo oxide run vecadd

# Or build without running
cargo oxide build vecadd

# Manual: pass the backend as a rustc flag
RUSTFLAGS="-Z codegen-backend=/path/to/librustc_codegen_cuda.so" cargo run --release
```

### Required Compiler Flags

These are set automatically by `cargo oxide`. For manual invocations, all three are required:

| Flag                                    | Why                                                 |
|-----------------------------------------|-----------------------------------------------------|
| `-C opt-level=3`                        | Enables MIR inlining and const-prop for device code |
| `-C debug-assertions=off`               | Strips debug checks from device code                |
| `-Z mir-enable-passes=-JumpThreading`   | Prevents barrier duplication across branches        |

> `panic=abort` is **not** required. The backend treats all unwind paths as unreachable since the CUDA toolchain does not support unwinding today (the hardware could; this is a compiler/runtime limitation).

### Environment Variables

| Variable                    | Effect                         |
|-----------------------------|--------------------------------|
| `CUDA_OXIDE_VERBOSE`        | Verbose compilation output     |
| `CUDA_OXIDE_PTX_DIR`        | Output directory for PTX files |
| `CUDA_OXIDE_TARGET`         | GPU architecture override      |
| `CUDA_OXIDE_LLC`            | Path to a specific `llc`       |
| `CUDA_OXIDE_DUMP_MIR`       | Dump the `dialect-mir` module  |
| `CUDA_OXIDE_DUMP_LLVM`      | Dump the LLVM dialect module   |
| `CUDA_OXIDE_SHOW_RUSTC_MIR` | Dump raw rustc MIR             |
| `CUDA_OXIDE_EMIT_NVVM_IR`   | Emit NVVM IR for libNVVM       |

`cargo oxide --arch <sm_XX>` sets `CUDA_OXIDE_TARGET`. When it is unset,
PTX output auto-detects the required target from generated LLVM IR.

## Source Layout

```text
src/
├── lib.rs              # CodegenBackend trait implementation (entry point)
├── collector.rs        # Device function discovery and call-graph walk
└── device_codegen.rs   # Bridge to the cuda-oxide MIR → PTX pipeline
```

## Examples

The `examples/` directory contains standalone kernel crates that exercise different features:

| Example                      | What it covers                                             |
|------------------------------|------------------------------------------------------------|
| `vecadd`                     | Basic vector addition -- the "hello world" kernel          |
| `generic`                    | Generic kernels (`scale<T>`)                               |
| `ord_cmp`                    | Device-side `Ord::cmp` for signed and unsigned integers    |
| `manual_launch_generic`      | Lower-level generic launch API regression                  |
| `cuda_module_contract`       | Typed launch ABI argument marshalling                      |
| `abi_hmm`                    | HMM pointers, struct layout, closures                      |
| `device_closures`            | Move and non-move closures passed to kernels               |
| `ref_index_projections`      | Borrow / raw-pointer address projections (issue #120)      |
| `ref_operand_mul`            | `Mul` impl on `&Foo` with `Output = Foo` (issue #133)      |
| `cross_crate_kernel`         | Kernels defined in a library crate                         |
| `async_vecadd`               | Async CUDA streams with `cuda-async`                       |
| `async_mlp`                  | Multi-layer perceptron using async streams                 |
| `sharedmem`                  | Shared memory usage                                        |
| `dynamic_smem`               | Dynamic shared memory allocation                           |
| `barrier`                    | `__syncthreads` and barrier semantics                      |
| `atomics`                    | Atomic operations on device                                |
| `atomic_f16`                 | Scalar f16 atomic correctness checks and microbenchmarks   |
| `printf`                     | Device-side `printf` via FFI                               |
| `tma_copy`                   | Tensor Memory Accelerator copies (Hopper+)                 |
| `tma_multicast`              | TMA with multicast across CTAs                             |
| `wgmma`                      | Warpgroup MMA (Hopper tensor cores)                        |
| `tcgen05` / `tcgen05_matmul` | 5th-gen tensor cores (Blackwell datacenter)                |
| `gemm` / `gemm_sol`          | GEMM implementations at various optimization levels        |
| `cluster`                    | Thread block clusters                                      |
| `clc`                        | Cluster Launch Control                                     |
| `warp_reduce`                | Warp-level reductions                                      |
| `cpp_consumes_rust_device`   | C++ host code consuming Rust-generated PTX                 |
| `device_ffi_test`            | Rust kernels calling external C++/CCCL functions via LTOIR |
| `mathdx_ffi_test`            | MathDx FFI: cuFFTDx thread-level FFT + cuBLASDx block GEMM |

Examples starting with `error_` are expected to fail. See [STATUS.md](STATUS.md)
for what each one tests and why it fails.

Run any example with:

```bash
cargo oxide run <example_name>

# See the full compilation pipeline
# (MIR → dialect-mir → LLVM dialect → LLVM IR → PTX)
cargo oxide pipeline <example_name>
```

## Key Design Decisions

- **Device code is `no_std`**. Functions reachable from a `#[kernel]` may only call into `core`, `cuda_device`, or the local crate. Use of `std` or `alloc` is a compile-time error.
- **Arguments are scalarized** at the host/device boundary. Aggregates (slices, structs) are flattened to scalars for the CUDA launch ABI and reconstructed inside the kernel. This is transparent to the user.
- **Struct layout matches rustc exactly**. Device-side structs use explicit padding derived from rustc's layout queries, so `#[repr(C)]` is not required.
- **Closures work**. Both `move` closures (capture by value) and non-move closures (capture by reference via HMM) can be passed to kernels.
- **Unwind paths are unreachable**. The backend ignores all MIR unwind edges, so `panic=abort` and custom sysroots are unnecessary. If a panic condition is hit at runtime, the GPU traps.

### Nightly Rust

The backend uses `#![feature(rustc_private)]` and pins to a specific nightly via `rust-toolchain.toml`. The workspace currently uses `nightly-2026-04-03`.
