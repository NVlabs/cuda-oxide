# cuda-core

Safe RAII wrappers around the CUDA driver API.

## Overview

This crate turns raw CUDA handles into Rust types with automatic cleanup on drop:

| Type                  | Wraps                  | Cleanup call                       |
|-----------------------|------------------------|------------------------------------|
| `CudaContext`         | `CUcontext` (primary)  | `cuDevicePrimaryCtxRelease_v2`     |
| `CudaStream`          | `CUstream`             | `cuStreamDestroy_v2`               |
| `CudaEvent`           | `CUevent`              | `cuEventDestroy_v2`                |
| `CudaModule`          | `CUmodule`             | `cuModuleUnload`                   |
| `CudaFunction`        | `CUfunction`           | (prevented from outliving module)  |
| `PinnedHostBuffer<T>` | pinned host memory     | `cuMemFreeHost`                    |

## Key APIs

- **Context**: `CudaContext::new(ordinal)` retains the primary context, binds it to the calling thread, and returns an `Arc<CudaContext>`.
- **Streams**: `ctx.new_stream()` creates a non-blocking stream. `stream.launch_host_function(f)` enqueues an `FnOnce` callback after all prior stream work completes -- this is the bridge to Rust futures in `cuda-async`.
- **Modules**: `ctx.load_module_from_file("kernel.ptx")` / `ctx.load_module_from_ptx_src(src)` load compiled GPU code. `module.load_function("kernel_name")` extracts a callable function handle.
- **Raw interop**: users who need it can access non-owning raw driver handles such as `CudaModule::cu_module` for libraries that need direct module access. These APIs are unsafe: keep the cuda-oxide owner alive, do not transfer or destroy the handle, and bind the owning context before raw driver calls.
- **Memory**: Async (`malloc_async`, `free_async`, `memcpy_htod_async`, ...) and sync (`malloc_sync`, `free_sync`) device memory operations.
- **Device buffers**: `DeviceBuffer<T>` owns device memory and provides host-device transfer helpers for `T: DeviceCopy`. Safe borrowed host-to-device copies synchronize before returning; unchecked async HtoD helpers are `unsafe`.
- **Pinned host memory**: `PinnedHostBuffer<T>` allocates page-locked host memory for faster transfers. The async transfer helpers (`DeviceBuffer::from_pinned_host`, `copy_from_pinned_host_async`, `copy_to_pinned_host_async`) are `unsafe` because they only enqueue the copy and the caller must keep the pinned buffer alive until `stream.synchronize()`. Use `copy_to_pinned_host` for a blocking DtoH helper that syncs internally.
- **Launch**: the `unsafe` `launch_kernel(func, grid, block, smem, stream, params)` family enqueues raw launches. The caller must prove both the argument ABI and the kernel's geometry/resource assumptions. `launch_kernel_ex(...)` adds cluster dimensions (Hopper+); `launch_kernel_cooperative(...)` enables `Grid::sync()` for grid-wide barriers.
- **Events**: `ctx.new_event(flags)` for synchronization points and GPU-side timing via `event.elapsed_ms(end)`.

## Usage

```rust
use cuda_core::{CudaContext, LaunchConfig};

let ctx = CudaContext::new(0)?;
let stream = ctx.new_stream()?;
let module = ctx.load_module_from_file("vecadd.ptx")?;
let func = module.load_function("vecadd")?;
// ... allocate memory, launch kernel, synchronize
```
