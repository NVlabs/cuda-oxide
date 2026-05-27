# Kernels and Device Functions

A **kernel** is a function that runs on the GPU -- the entry point that the host
launches across thousands of threads. A **device function** is a helper that runs
on the GPU but can only be called from another device function or kernel, never
from the host. This chapter covers both, along with the Rust patterns that are
(and aren't) supported in device code.

:::{seealso}
[CUDA Programming Guide -- Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/#kernels)
for the authoritative CUDA C++ reference on kernel and device functions.
:::

## `#[kernel]` -- the GPU entry point

Annotating a function with `#[kernel]` tells cuda-oxide to compile it as a GPU
entry point. The function must return `()` -- kernels communicate results by
writing to output buffers, not by returning values.

```rust
use cuda_device::{kernel, thread, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(c_elem) = c.get_mut(idx) {
        *c_elem = a[idx.get()] + b[idx.get()];
    }
}
```

Under the hood, `#[kernel]` does three things:

1. **Renames** the function into the reserved `cuda_oxide_kernel_<hash>_<name>`
   namespace so the compiler's collector can identify it as a device entry
   point. The exact prefix is owned by the workspace-internal
   `reserved-oxide-symbols` crate; the `<hash>` suffix makes the namespace
   unguessable for user code.
2. **Adds `#[no_mangle]`** to preserve the symbol name in the generated PTX.
3. **Generates a marker struct** implementing `CudaKernel` (or
   `GenericCudaKernel` for generic kernels) so host launch code can look up the
   correct PTX entry point at compile time.

In the generated PTX, a kernel becomes a `.entry` directive -- the GPU
equivalent of `main`:

```text
.entry vecadd(.param .u64 a, .param .u64 a_len, ...) { ... }
```

### Parameter constraints

Kernel parameters cross the host/device ABI boundary through
**argument scalarization** (covered in the
[Memory and Data Movement](memory-and-data-movement.md) chapter). The key
rules:

- **Slices** (`&[T]`, `DisjointSlice<T>`) become a pointer + length pair.
- **Scalars** (`u32`, `f32`, etc.) are passed directly.
- **Structs and closures by value** travel as a single byval `.param`. The
  field-by-field flattening still applies to internal device-to-device
  calls, but the kernel boundary itself receives the whole aggregate as
  one value to match the single packet slot the host launcher pushes.
- **No heap-allocated types** (`Vec`, `String`, `Box`) -- the `alloc` crate is
  allowed through the compiler, but no device-side `#[global_allocator]` is
  configured today. Even with one, device `malloc` is extremely slow.

## Device helper functions

Not all GPU code belongs in the kernel itself. You can factor logic into helper
functions that the compiler will also compile for the GPU.

### Auto-discovered helpers

The simplest approach: just write a normal Rust function and call it from your
kernel. The compiler's **collector** traverses the call graph from each
`#[kernel]` entry point and automatically compiles every reachable function for
the GPU -- no annotation needed:

```rust
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    if x < lo { lo } else if x > hi { hi } else { x }
}

#[kernel]
pub fn apply_clamp(input: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = clamp(input[idx.get()], 0.0, 1.0);
    }
}
```

The `clamp` function is compiled to a PTX `.func` (device function) and
typically inlined by the compiler, so there is no call overhead.

### When `#[device]` is needed

The `#[device]` attribute is required in three specific scenarios where
auto-discovery is not sufficient:

| Scenario                         | Why `#[device]` is needed                                                             |
|:---------------------------------|:--------------------------------------------------------------------------------------|
| **Standalone device libraries**  | No `#[kernel]` in the crate, so the collector has no entry point to walk from         |
| **Cross-crate device functions** | The function is in a different crate from the kernel                                  |
| **Device FFI**                   | The function is exposed as `#[device] extern "C"` for linking with CUDA C++ via LTOIR |

```rust
use cuda_device::device;

#[device]
pub fn magnitude(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}
```

### `#[kernel]` vs `#[device]`

| Feature                  | `#[kernel]`             | `#[device]`                         | Auto-discovered        |
|:-------------------------|:------------------------|:------------------------------------|:-----------------------|
| PTX directive            | `.entry`                | `.func`                             | `.func` (or inlined)   |
| Launchable from host     | Yes, via typed module   | No                                  | No                     |
| Can return a value       | No (must be `()`)       | Yes                                 | Yes                    |
| Callable from device code| Yes                     | Yes                                 | Yes                    |
| Annotation required      | Always                  | Only for standalone/cross-crate/FFI | Never                  |

## What Rust works on the GPU

cuda-oxide compiles standard Rust through `rustc` -- it is not a subset
language. That said, GPU code runs in a `no_std` environment without a
device-side heap allocator configured, so certain Rust features are
unavailable today. Here is the current support matrix:

### Supported

| Feature                                                | Notes                                          |
|:-------------------------------------------------------|:-----------------------------------------------|
| Primitive types (`u8`..`u64`, `f32`, `f64`, `bool`)    | Full support                                   |
| Structs and tuples                                     | Decomposed at ABI boundary                     |
| Enums (`Option<T>`, `Result<T,E>`, custom)             | Including `match`                              |
| `match` / `if` / `if let`                              | Multi-way branching                            |
| `for` loops and `while` loops                          | Range-based and iterator-based                 |
| Iterators (`.iter()`, `.enumerate()`)                  | Desugared through MIR                          |
| `break` and `continue`                                 | Inside loops                                   |
| Arrays (`[T; N]`)                                      | Read, write, indexing                          |
| Slices (`&[T]`)                                        | Read-only; mutable writes via `DisjointSlice`  |
| Closures (within device code)                          | Normal Rust semantics                          |
| Generic functions                                      | Monomorphized per call site                    |
| `unsafe` blocks and raw pointers                       | For advanced patterns                          |

### Not supported

| Feature                           | Reason                                                              | Alternative                          |
|:----------------------------------|:--------------------------------------------------------------------|:-------------------------------------|
| `String`, `Vec`, `Box`            | Require heap allocator (no device-side `#[global_allocator]` today) | Use fixed-size arrays or slices      |
| `format!`, `println!`             | Require formatting machinery + I/O                                  | Use `gpu_printf!`                    |
| `std` I/O, networking, filesystem | No OS on GPU                                                        | Communicate via buffers              |
| Trait objects (`dyn Trait`)       | Require vtable dispatch                                             | Use generics (monomorphized)         |
| `panic!` with message             | Formatting + allocation                                             | Use `gpu_assert!` or `debug::trap()` |

:::{tip}
If you accidentally use an unsupported feature, the compiler will produce a
clear error: `"CUDA-OXIDE: FORBIDDEN CRATE IN DEVICE CODE"` with a list of
allowed crates (`core`, `alloc`, `cuda_device`, and your local crate).
:::

## `#[launch_bounds]` -- occupancy hints

The `#[launch_bounds]` attribute tells the compiler how many threads you intend
to launch per block. This lets the PTX assembler make better register allocation
decisions and can improve occupancy:

```rust
#[kernel]
#[launch_bounds(256, 2)]
pub fn optimized_kernel(mut out: DisjointSlice<f32>) {
    // ...
}
```

| Parameter      | Required | PTX directive   | Description                      |
|:---------------|:---------|:----------------|:---------------------------------|
| `max_threads`  | Yes      | `.maxntid`      | Maximum threads per block        |
| `min_blocks`   | No       | `.minnctapersm` | Minimum concurrent blocks per SM |

The generated PTX includes these directives:

```text
.entry optimized_kernel .maxntid 256, 1, 1 .minnctapersm 2 { ... }
```

:::{tip}
`#[launch_bounds]` must appear **after** `#[kernel]`:

```rust
#[kernel]
#[launch_bounds(256, 2)]   // correct
pub fn my_kernel(...) { }
```

:::

## The collector -- how device code is discovered

When you build with `cargo oxide`, the `rustc-codegen-cuda` backend runs a
**collector** pass that determines which functions to compile for the GPU:

1. Scan all compilation units for functions in the reserved
   `cuda_oxide_kernel_<hash>_` namespace (generated by `#[kernel]`).
2. For each kernel, **traverse the call graph** and collect all transitively
   reachable functions.
3. **Filter** each callee against the allowed-crate list:

| Crate            | Allowed | Why                                                                                        |
|:-----------------|:--------|:-------------------------------------------------------------------------------------------|
| Your local crate | Yes     | Your kernel and helper code                                                                |
| `cuda_device`    | Yes     | GPU intrinsics (threads, warps, shared memory)                                             |
| `core`           | Yes     | `no_std` Rust core library                                                                 |
| `std`            | No      | Requires OS facilities not available on GPU                                                |
| `alloc`          | Allowed | Passes the collector, but no device-side allocator is wired up yet. Link-time error today. |

If the collector encounters a call into a forbidden crate, it reports a
compile-time error rather than generating broken PTX.

```{figure} images/collector-traversal.svg
:align: center
:width: 100%

The device code collector: starting from #[kernel] entry points, the compiler
walks the call graph to discover all reachable device functions, then filters
each callee against the allowed-crate list (local crate, cuda_device, core).
The output is a PTX module with .entry and .func directives.
```

## `no_std` and panic behavior

Device code runs in an implicit `#![no_std]` environment. You do not need to add
this attribute yourself -- the compiler backend handles it.

**Panic behavior:** all unwind paths in MIR are treated as unreachable. If a
panic actually triggers at runtime (e.g., an array bounds check fails), the GPU
executes a **trap instruction**, which causes the host to receive
`CUDA_ERROR_ILLEGAL_INSTRUCTION`. This is semantically equivalent to
`panic=abort` but does not require any special compiler flags.

In practice this means:

- `unwrap()` and `expect()` work but will trap the GPU on `None`/`Err`.
- `assert!` and `debug_assert!` work but trap on failure.
- `panic!("message")` is **not** supported (the formatting machinery is
  unavailable) -- use `gpu_assert!` or `debug::trap()` instead.

:::{seealso}
The [Error Handling and Debugging](error-handling-and-debugging.md) chapter
covers `gpu_printf!`, `gpu_assert!`, and `cargo oxide debug` for diagnosing
kernel failures.
:::
