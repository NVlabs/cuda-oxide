# The cuda-oxide Book

```{image} _static/images/logo.png
:alt: cuda-oxide logo
:align: center
:width: 780px
:class: mb-4
```

**cuda-oxide** is an experimental Rust-to-CUDA compiler that lets you write (SIMT) GPU kernels in safe(ish), idiomatic Rust. It compiles standard Rust code directly to PTX — no DSLs, no foreign language bindings, just Rust.

:::{note}
This book assumes familiarity with the Rust programming language, including ownership, traits, and generics. Later chapters on async GPU programming also assume working knowledge of `async`/`.await` and runtimes like tokio.

For a refresher, see [The Rust Programming Language](https://doc.rust-lang.org/book/), [Rust by Example](https://doc.rust-lang.org/rust-by-example/), or the [Async Book](https://rust-lang.github.io/async-book/).
:::

---

## Project Status

The v0.1.0 release is an early-stage alpha: **expect bugs, incomplete features, and API breakage** as we work to improve it. We hope you'll try it and help shape its direction by sharing feedback on your experience.

---

## 🚀 Quick start

```rust
use cuda_device::{cuda_module, kernel, thread, DisjointSlice};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        let i = idx.get();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[i] + b[i];
        }
    }
}

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).unwrap();

    let a = DeviceBuffer::from_host(&stream, &[1.0f32; 1024]).unwrap();
    let b = DeviceBuffer::from_host(&stream, &[2.0f32; 1024]).unwrap();
    let mut c = DeviceBuffer::<f32>::zeroed(&stream, 1024).unwrap();

    module
        .vecadd(&stream, LaunchConfig::for_num_elems(1024), &a, &b, &mut c)
        .unwrap();

    let result = c.to_host_vec(&stream).unwrap();
    assert_eq!(result[0], 3.0);
}
```

Build and run with `cargo oxide run vecadd` upon installing the [prerequisites](getting-started/installation.md).

:::{note}
`#[cuda_module]` embeds the generated device artifact into the host binary and
generates a typed `kernels::load` function plus one launch method per kernel.
The lower-level `load_kernel_module` and unsafe `cuda_launch!` APIs remain
available when you need to load a specific sidecar artifact or build custom
launch code.
:::

---

## Why cuda-oxide?

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🦀  Rust on the GPU
Write GPU kernels with Rust's type system and ownership model.
Safety is a first-class goal, but GPUs have subtleties — read about
[the safety model](gpu-safety/the-safety-model.md).
:::

:::{grid-item-card} 💎  A SIMT Compiler
Not a DSL. A custom rustc codegen backend that compiles
pure Rust to PTX.
:::

:::{grid-item-card} ⚡  Async Execution
Compose GPU work as lazy `DeviceOperation` graphs.
Schedule across stream pools. Await results with `.await`.
:::

::::

```{toctree}
:hidden:
:maxdepth: 2
:caption: Getting Started

getting-started/installation
getting-started/hello-gpu
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Writing GPU Programs

gpu-programming/execution-model
gpu-programming/kernels-and-device-functions
gpu-programming/memory-and-data-movement
gpu-programming/launching-kernels
gpu-programming/closures-and-generics
gpu-programming/error-handling-and-debugging
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Safety on the GPU

gpu-safety/the-safety-model
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Async GPU Programming

async-programming/the-device-operation-model
async-programming/combinators-and-composition
async-programming/scheduling-and-streams
async-programming/concurrent-execution
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Building a Real Application

projects/async-mlp-pipeline
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Advanced GPU Features

advanced/shared-memory-and-synchronization
advanced/warp-level-programming
advanced/tensor-memory-accelerator
advanced/matrix-multiply-accelerators
advanced/cluster-programming
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Inside the Compiler

compiler/architecture-overview
compiler/pliron
compiler/rustc-public
compiler/rustc-codegen-cuda
compiler/mir-importer
compiler/mlir-dialects
compiler/lowering-pipeline
compiler/adding-new-intrinsics
compiler/fuzzing-and-differential-testing
```

```{toctree}
:hidden:
:maxdepth: 1
:caption: Appendix

appendix/building-from-source
appendix/api-quick-reference
appendix/supported-features
appendix/cuda-cpp-comparison
appendix/ecosystem
appendix/glossary
```
