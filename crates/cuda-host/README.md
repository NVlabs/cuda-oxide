# cuda-host

Host-side infrastructure for typed CUDA module loading and kernel launches in
cuda-oxide.

The primary interface is `#[cuda_module]`. Place kernels in an inline module,
keep `#[kernel]` on the actual GPU entry points, then load the embedded device
artifact as a typed Rust value.

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
        let idx = thread::index_1d();
        if let Some(c_elem) = c.get_mut(idx) {
            *c_elem = a[idx.get()] + b[idx.get()];
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();

    const N: usize = 1024;
    let a_dev = DeviceBuffer::from_host(&stream, &vec![1.0f32; N])?;
    let b_dev = DeviceBuffer::from_host(&stream, &vec![2.0f32; N])?;
    let mut c_dev = DeviceBuffer::<f32>::zeroed(&stream, N)?;

    let module = kernels::load(&ctx)?;
    module.vecadd(
        &stream,
        LaunchConfig::for_num_elems(N as u32),
        &a_dev,
        &b_dev,
        &mut c_dev,
    )?;

    Ok(())
}
```

## Generated API

`#[cuda_module]` adds these items to the annotated module:

| Item | Purpose |
|------|---------|
| `LoadedModule` | Typed handle around the embedded CUDA module and cached kernel functions |
| `load(&Arc<CudaContext>)` | Load the current package's embedded bundle; unsafe when the module has a launch contract |
| `load_named(&Arc<CudaContext>, name)` | Load a specific embedded bundle by name; unsafe when the module has a launch contract |
| `from_module(Arc<CudaModule>)` | Wrap an already-loaded CUDA module; unsafe when the module has a launch contract |
| `LoadedModule::{kernel}` | One launch method per `#[kernel]` function |
| `load_async(device_id)` | With feature `async`, load from a `cuda-async` device context; unsafe when contracted |
| `LoadedModule::{kernel}_async` | With feature `async`, build a lazy `AsyncKernelLaunch` |
| `LoadedModule::{kernel}_async_owned` | With feature `async`, build an owned async launch that returns its buffers |

Kernel parameters are mapped into host launch parameters:

| Kernel parameter | Host method parameter |
|------------------|-----------------------|
| `&[T]` | `&DeviceBuffer<T>` |
| `&mut [T]` | `&mut DeviceBuffer<T>` |
| `DisjointSlice<T>` | `&mut DeviceBuffer<T>` |
| `Copy` scalar, struct, closure, or raw pointer | unchanged |

Because the launches are ordinary methods, rust-analyzer and rustc can complete
kernel names, show argument names, and type-check arguments before the program
runs. By-value arguments are copied into the CUDA launch packet through the
`KernelScalar` boundary; device slices are encoded as pointer-plus-length pairs.

## Prepared Launch Contracts

`#[launch_contract]` is an opt-in path for kernels whose geometry and resource
assumptions are part of correctness, not merely a tuning choice:

```rust
#[kernel]
#[launch_bounds(256)]
#[launch_contract(
    domain = 1,
    block = (256, 1, 1),
    dynamic_shared = 1024,
    dynamic_shared_alignment = 128,
    min_compute_capability = (9, 0),
)]
fn reduce(input: &[f32], mut output: DisjointSlice<f32>) { /* ... */ }
```

Preparation resolves the exact generic kernel entry and checks the live CUDA
device and function once:

```text
LaunchConfig1D
      │ check block/grid limits, shared memory, CC, cluster/cooperative rules
      ▼
PreparedLaunch<reduce<T>>
      │ immutable function + configuration + context
      └──────────────> repeated launches make no capability/resource queries
```

```rust
let prepared = module.prepare_reduce(LaunchConfig1D::new(blocks, 256, 1024))?;
module.reduce(&stream, &prepared, &input, &mut output)?;
```

`LaunchConfig1D`, `LaunchConfig2D`, and `LaunchConfig3D` have private fields;
a 2-D configuration cannot be passed to a 1-D contract. The prepared value is
also branded with the exact kernel specialization, so `reduce::<f32>` and
`reduce::<f64>` are not interchangeable. Generic closures can use the generated
`prepare_{kernel}_for(&closure, config)` helper to infer their anonymous type.

For contracted kernels, raw `LaunchConfig` is available only through generated
unsafe methods such as `reduce_unchecked`. Borrowed and owned async methods
return immutable prepared wrappers, so safe code cannot change their geometry
after validation; they recheck the scheduler-selected stream's context at
submission time.

Binding code is a one-time unsafe boundary for contracted modules. Embedded
bundles are currently named per package, not per library/binary target, so even
`load()` cannot prove that a same-package sibling contains the matching ABI and
contract. `load`, `load_async`, `from_module`, `load_named`, and
`load_async_named` therefore require the caller to assert artifact provenance.
For generic modules, the caller must also ensure the merged PTX bundles contain
the matching specializations and no conflicting entry definitions. Preparation
and repeated launches are safe after that binding.

The declared dynamic shared-memory byte range is an author contract because
arbitrary pointer offsets cannot be inferred. Alignment is compiler-enforced:
the marker emitted by `#[launch_contract]` is merged with alignment requests in
the body and reachable local helpers, and the stronger value reaches PTX.
Prelinked external helpers keep the alignment recorded when they were compiled.

Cluster and cooperative contracts are validated separately. A contract that
combines both currently fails preparation because the available occupancy query
cannot prove the combined residency rule; the unsafe launch method remains the
explicit expert escape hatch.

This provides the typed host half of issue #115: an opted-in 1-D kernel cannot
be launched with Y/Z geometry through safe generated methods. Connecting that
host proof to every `ThreadIndex` created through transitive device helpers is
separate work; uncontracted kernels keep their existing API.

Enable the `async` feature to generate async launch methods. They use the same
scalar mapping, but take no stream argument:

```rust
use cuda_async::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
module
    .vecadd_async(LaunchConfig::for_num_elems(N as u32), &a_dev, &b_dev, &mut c_dev)?
    .sync()?;
```

For async launches, device-slice parameters accept either `DeviceBuffer<T>` or
`cuda_async::device_box::DeviceBox<[T]>`. Borrowed methods return
`AsyncKernelLaunch<'_>`, so Rust keeps referenced buffers and non-`'static`
scalar arguments borrowed until the lazy operation is dropped, `.sync()` has
returned, or `.await` has completed.

Use `{kernel}_async_owned` when the operation needs to leave the current stack
frame, for example in a spawned Tokio task or a long-lived pipeline:

```rust
let (a_dev, b_dev, c_dev) = module
    .vecadd_async_owned(LaunchConfig::for_num_elems(N as u32), a_dev, b_dev, c_dev)?
    .await?;
```

Owned async launch methods take device-slice arguments by value, keep them alive
for the GPU work, and return them as the operation output after completion.
Scalar arguments for owned async launches must be `'static`.

## Lower-Level Pieces

`CudaKernel` and `GenericCudaKernel` remain the marker traits generated by
`#[kernel]` and used by the typed loader. `cuda_launch!` remains available as
the unsafe low-level escape hatch for modules loaded at runtime by name; it
cannot check argument count or types against the kernel, so every use must be
wrapped in `unsafe { }`. For kernels embedded in your own crate, use
`#[cuda_module]` instead: it generates typed launch methods from the kernel
signatures.

`cuda_launch_async!` is also lower-level. It can describe lazy work from raw
device pointers, so callers must ensure the pointed-to allocations outlive the
operation. Generated borrowed async methods encode that requirement as Rust
borrows, and generated owned async methods move buffers into the operation for
spawned tasks.

`#[cuda_module]` loads the artifact produced by the compiler. PTX and cubin
payloads load directly. NVVM IR and LTOIR normally compile for the target
recorded with the artifact. A module built for a standard pre-Blackwell target,
such as `sm_86`, can also be converted to PTX and JIT-compiled by the CUDA
driver on Blackwell. This forward path is not available for suffixed targets,
such as `sm_90a`, and artifacts built for newer GPUs cannot run on older GPUs.
Because this path JIT-compiles PTX, the installed driver must support the PTX
version produced by the selected CUDA toolkit.

Pre-Blackwell targets use typed-pointer NVVM IR. Blackwell and newer targets
use opaque-pointer NVVM IR. The compiler records the selected target in the
embedded bundle and in the `<module>.target` file used by the lower-level
loader.

Older artifacts without a recorded target must be rebuilt or loaded with
`CUDA_OXIDE_TARGET` set to their original target.

File-backed NVVM IR and LTOIR cache native cubins below
`.oxide-artifacts/ltoir-cubin-cache/v1`. A cache entry is reused only when the
source, target, module names, ordered options, libdevice, and exact loaded
libNVVM/nvJitLink binaries all match. The cubin and optional LTOIR are verified
and published together, so a stopped or concurrent build cannot mix outputs.
If either CUDA library cannot be identified exactly, cuda-oxide rebuilds. PTX
selection and the pre-Blackwell to Blackwell PTX bridge do not use this cache.
The first compiler/linker handles are retained for the process lifetime;
restart the process to select another toolkit or after replacing one in place.
Remove the `.oxide-artifacts` directory to clear all cached entries.

## Tiling Utilities (tcgen05)

Host-side layout transformations for Blackwell tensor cores. tcgen05 requires
specific 8x8 tile arrangements:

| Function | Description |
|----------|-------------|
| `to_k_major_f16` | Row-major to tcgen05 K-major, matrix A |
| `to_mn_major_f16` | Row-major to tcgen05 MN-major, matrix B |
| `k_major_index` | Compute linear index in K-major layout |
| `mn_major_index` | Compute linear index in MN-major layout |
| `print_layout_indices` | Debug print layout as 2D table |
| `TILE_SIZE` | Constant `8` |

## Further Reading

- [cuda-device](../cuda-device/) -- device-side intrinsics
- [cuda-macros](../cuda-macros/) -- proc-macro implementations
- [cuda-core](../cuda-core/) -- CUDA driver API, `DeviceBuffer`, `LaunchConfig`
- [cuda-async](../cuda-async/) -- async scheduling
