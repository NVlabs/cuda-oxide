# cuda-macros

Procedural macros for writing CUDA kernels in Rust. Provides `#[cuda_module]`
for typed embedded-module loading, `#[kernel]` for GPU entry points,
`#[device]`, `#[launch_bounds]`, `#[cluster_launch]`, `gpu_printf!`, and the
lower-level `cuda_launch!` / `cuda_launch_async!` migration macros.

## Attributes

### `#[kernel]` -- GPU Kernel Entry Point

Marks a function as a CUDA kernel. Generates:
1. An entry point renamed into the reserved `cuda_oxide_kernel_<hash>_<name>` namespace
   (with `#[no_mangle]`) so the codegen backend can find it. The hash makes the prefix
   unguessable for user code; see `crates/reserved-oxide-symbols/` for the contract.
2. A `__<name>_CudaKernel` marker struct implementing `CudaKernel` (or `GenericCudaKernel` for generics).

> **Reserved names.** The macros refuse to compile any function whose name starts with
> `cuda_oxide_` -- that namespace is reserved for cuda-oxide-internal mangling. The check
> is enforced at expansion time so the error points at the offending source line.

```rust
use cuda_device::{kernel, DisjointSlice};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

**Generic kernels** work in two modes:

```rust
// Mode 1: call-site instantiation (PTX name from type_name)
#[kernel]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) { ... }
// Launch: module.scale::<f32>(&stream, config, factor, &input, &mut out)?

// Mode 2: explicit instantiation list
#[kernel(f32, i32)]
pub fn scale<T: Copy + Mul<Output = T>>(factor: T, input: &[T], mut out: DisjointSlice<T>) { ... }
// Generates named entry points: scale_f32, scale_i32
```

### `#[device]` -- Device Helper Functions and Externs

Device functions run on GPU but are not entry points. Works on both regular functions and `extern "C"` blocks:

```rust
#[device]
pub fn magnitude(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}

// Extern device functions (e.g. from libdevice or cuBLASDx)
#[device]
extern "C" {
    fn __nv_expf(x: f32) -> f32;
}
```

| Feature              | `#[kernel]`          | `#[device]`          |
|----------------------|----------------------|----------------------|
| Entry point          | Yes (PTX `.entry`)   | No (PTX `.func`)     |
| Can return values    | No (must be `()`)    | Yes                  |
| Callable from host   | Via `#[cuda_module]` | No                   |
| Callable from device | Yes                  | Yes                  |

### `#[launch_bounds(max_threads, min_blocks)]`

Occupancy hints for register allocation. Must come **after** `#[kernel]`.

```rust
#[kernel]
#[launch_bounds(256, 2)]  // max 256 threads, min 2 blocks per SM
pub fn optimized(out: DisjointSlice<f32>) { ... }
// PTX: .entry optimized .maxntid 256 .minnctapersm 2 { ... }
```

### `#[cluster_launch(x, y, z)]`

Compile-time thread block cluster dimensions (Hopper+). Must come **after** `#[kernel]`.

```rust
#[kernel]
#[cluster_launch(4, 1, 1)]  // 4 blocks per cluster
pub fn cluster_kernel(out: DisjointSlice<u32>) { ... }
// PTX: .entry cluster_kernel .reqnctapercluster 4, 1, 1 { ... }
```

### `#[convergent]`, `#[pure]`, `#[readonly]`

Semantic markers for the codegen backend (pass-through -- no code transformation).

## `#[cuda_module]` -- Typed Embedded Module Loading

Wrap an inline module containing `#[kernel]` functions to generate a typed
loader and per-kernel launch methods:

```rust
#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { ... }
}

let module = kernels::load(&ctx)?;
module.vecadd(&stream, LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?;
```

When `cuda-host` is built with its `async` feature, async code can load the
same embedded module from a `cuda-async` device context:

```rust
let module = kernels::load_async(0)?;
module.vecadd_async(LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?.sync()?;
```

Borrowed async methods return `AsyncKernelLaunch<'_>` and tie the lazy operation
to referenced buffers and borrowed scalar arguments. Owned async methods take
device buffers by value and return them after completion:

```rust
let (a_dev, b_dev, c_dev) = module
    .vecadd_async_owned(LaunchConfig::for_num_elems(N as u32), a_dev, b_dev, c_dev)?
    .await?;
```

## `cuda_launch!` -- Lower-Level Synchronous Kernel Launch

```rust
cuda_launch! {
    kernel: vecadd,                                  // or scale::<f32> for generics
    stream: stream,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    cluster_dim: (4, 1, 1),                          // optional, uses launch_kernel_ex
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
}
```

### Argument Forms

| Syntax                | Kernel Parameter    | Marshalling                         |
|-----------------------|---------------------|-------------------------------------|
| `expr`                | `T` (scalar)        | `&mut value` as `*mut c_void`       |
| `slice(buf)`          | `&[T]`              | Device pointer + length (two args)  |
| `slice_mut(buf)`      | `DisjointSlice<T>`  | Device pointer + length (two args)  |
| `move \|..\| body`    | Closure `F`         | Each capture by value               |
| `\|..\| body`         | Closure `F`         | Pointers to captures (HMM)          |

### PTX Name Resolution

| Kernel Kind   | PTX Name                                                |
|:--------------|:--------------------------------------------------------|
| Non-generic   | Original function name (`vecadd`)                       |
| Generic       | `{name}_TID_{hex32}` (fixed length regardless of arity) |
| Closure-only  | Same as Generic — closure type is in the hashed tuple   |

`{hex32}` is rustc's stable 128-bit type-id hash for the *tuple* of
generic arguments `(T0, T1, ...)`, rendered as 32 lowercase hex chars.
The backend computes it via
`tcx.type_id_hash(Ty::new_tup(tcx, &args)).as_u128()`; the host computes
the same value via `cuda_host::type_id_u128::<(T0, T1, ...,)>()`. Both
sides share a single rustc invocation and go through the same
`erase_and_anonymize_regions` + stable-hash pipeline, so the strings
match byte-for-byte. Hashing the tuple (not each arg separately) keeps
the on-wire name a fixed `base.len() + 37` chars regardless of how many
generic parameters the kernel takes.

For generics, the macro forces monomorphization with a volatile pointer
trick so the kernel appears in the codegen unit even without a host-side
call.

## `cuda_launch_async!` -- Lower-Level Async Kernel Launch

Returns an `AsyncKernelLaunch` implementing `DeviceOperation` for `cuda-async` scheduling. Same argument forms as `cuda_launch!` but no `stream:` or `cluster_dim:` fields.

```rust
let op = cuda_launch_async! {
    kernel: vecadd,
    module: module,
    config: LaunchConfig::for_num_elems(N as u32),
    args: [slice(a_dev), slice(b_dev), slice_mut(c_dev)]
};
```

This is a lower-level API. Prefer `#[cuda_module]`'s borrowed async methods for
stack-local use and owned async methods for spawned tasks:

```text
raw pointer async:
  op stores only a device address
  owner can be dropped before op runs

typed borrowed async:
  op borrows buffers until completion

typed owned async:
  op owns buffers and returns them after completion
```

## `gpu_printf!` -- Device-Side Printf

Compiles to CUDA's `vprintf` with C vararg promotion rules. Format string must use C-style specifiers.

```rust
gpu_printf!("thread %d: val = %f\n", tid as i32, val as f64);
```

## Source Layout

```text
src/
├── lib.rs       # All proc-macro definitions (kernel, device, launch, etc.)
└── printf.rs    # gpu_printf! implementation
```

## Further Reading

- [cuda-device](../cuda-device/) -- re-exports these macros for convenience
- [cuda-host](../cuda-host/) -- `CudaKernel` / `GenericCudaKernel` traits used by generated code
- [cuda-core](../cuda-core/) -- `launch_kernel` / `launch_kernel_ex` called by generated code
