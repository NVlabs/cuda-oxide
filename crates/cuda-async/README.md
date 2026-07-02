# cuda-async

Async execution layer for CUDA device operations, built on top of `cuda-core`.

## Architecture

The crate is organized around a single idea: GPU work is described lazily, scheduled late, and composed freely before any hardware instruction is issued.

```text
  module.kernel_async(...)
         |
         v
  AsyncKernelLaunch          <-- immutable lazy operation, no GPU work yet
         |
    .and_then(|()| ...)      <-- compose with other DeviceOperations
         |
    .sync() / .await         <-- SchedulingPolicy picks a stream, executes
         |
         v
  cuLaunchKernel(stream)     <-- actual GPU dispatch
  cuLaunchHostFunc(stream)   <-- host callback wakes the Rust future
```

## Core concepts

### `DeviceOperation` trait

A lazy, composable unit of GPU work. Implements `Send + Sized + IntoFuture`. Key methods:

| Method              | Stream chosen by                      | Blocks thread? |
|---------------------|---------------------------------------|----------------|
| `.await`            | Default device's `SchedulingPolicy`   | No (suspends)  |
| `.sync()`           | Default device's `SchedulingPolicy`   | Yes            |
| `.sync_on(&stream)` | The explicit stream you provide       | Yes            |

Combinators: `.and_then(f)`, `.and_then_with_context(f)`, `.arc()`, `zip!(a, b)`, `unzip!(op)`.

### `DeviceFuture`

Bridges CUDA stream completion to Rust's `Future` trait. When a `DeviceOperation` is scheduled, `DeviceFuture` enqueues the work on a CUDA stream, then registers a host callback via `cuLaunchHostFunc` that wakes the async task when the GPU finishes.

### `SchedulingPolicy`

Determines which CUDA stream a `DeviceOperation` runs on:

- **`StreamPoolRoundRobin`** (default) -- rotates through a pool of N streams, enabling automatic overlap of independent operations.
- **`SingleStream`** -- all operations execute on one stream in strict FIFO order.

### `DeviceBox<T>`

Owning smart pointer for device memory. Frees memory asynchronously via `cuMemFreeAsync` on a dedicated deallocator stream when dropped, avoiding the full device synchronization that `cuMemFree` would cause.

### `AsyncDeviceContext`

Thread-local per-device state: CUDA context, scheduling policy, deallocator stream, and a kernel function cache. Initialized via `init_device_contexts(default_device_id, num_devices)`.

## Usage

Borrow buffers when the launch completes in the current scope:

```rust
use cuda_async::device_context::init_device_contexts;
use cuda_async::device_operation::DeviceOperation;
use cuda_host::cuda_module;
use cuda_core::LaunchConfig1D;

// 1. Enable cuda-host's "async" feature, then initialize once per thread.
init_device_contexts(0, 1)?;
// SAFETY: this program loads the embedded artifact generated from this exact
// contracted module, so its entry points and contract metadata match.
let module = unsafe { kernels::load_async(0)? };

// 2. Prove the launch matches vecadd's declared 1-D contract, then build it.
let prepared = module.prepare_vecadd(LaunchConfig1D::new(4, 256, 0))?;
let op = module.vecadd_async(&prepared, &a_dev, &b_dev, &mut c_dev);

// 3. Execute
op.sync()?;       // blocking
// or: op.await?  // async
```

Move buffers into the launch when it needs to be spawned or stored as a
`'static` future:

```rust
use std::future::IntoFuture;

let prepared = module.prepare_vecadd(LaunchConfig1D::new(4, 256, 0))?;
let op = module.vecadd_async_owned(&prepared, a_dev, b_dev, c_dev);

let (a_dev, b_dev, c_dev) = tokio::spawn(op.into_future()).await??;
```

The owned form keeps device buffers alive until the CUDA stream reaches the
kernel completion callback, then returns those buffers as the operation output.

## Raw launch safety boundary

The low-level builder is safe because it only stores inert data. A raw
`LaunchConfig` becomes runnable only through an explicit unsafe transition:

```text
AsyncKernelLaunchBuilder
       |  unsafe finalize_unchecked(raw_config)
       v
AsyncKernelLaunch          <-- DeviceOperation + IntoFuture
```

```rust,ignore
let mut builder = AsyncKernelLaunchBuilder::new(function);
builder.push_args((input_ptr, output_ptr, len));

// SAFETY: this kernel uses 1-D indexing, accepts 256 threads per block, and
// every pointer and argument matches its device-side signature.
let operation = unsafe { builder.finalize_unchecked(config) };
operation.sync()?;
```

Prefer generated prepared-launch methods. They validate the kernel's declared
contract and keep this unsafe proof inside generated code.

## Cancellation and deferred reclamation

Dropping a `DeviceFuture` never cancels GPU work. Once a kernel is submitted
it runs to completion no matter what the host does; cancellation only decides
*when the host releases* the resources the kernel is still using.

Dropping an in-flight future records a CUDA event on its assigned stream and
parks the stored result in a process-wide limbo. The drop itself never blocks
on GPU progress.

```text
drop(future)                      later sweep (any poll or drop)
  record event on stream            event passed?  -> drop the result
  park (event, result)              still running? -> keep it parked
```

Parked results are swept opportunistically whenever any `DeviceFuture` is
polled or dropped, and a result is only dropped once `cuEventQuery` proves
the device timeline passed its event. `cuda_async::reclaim::drain()` performs
a blocking drain when deterministic reclamation is needed (for example at the
end of a test); entries still parked at process exit are leaked and reclaimed
by the driver at teardown.

The same deferred path covers the case where the kernel launch succeeds but
host-callback registration fails: the owned resources are parked, not dropped,
even though the future resolves with an error.

Two failure endgames exist, both biased toward leaking rather than freeing
memory the device may still write to: if the completion event cannot be
recorded, the drop falls back to synchronizing the stream, and only when even
that fails is the result deliberately leaked with a message on stderr.

## Buffer lifetime safety

Async launches are lazy: building the operation does not enqueue GPU work. That
makes raw pointer launches easy to misuse because a `CUdeviceptr` is just an
integer handle with no Rust lifetime attached.

```text
raw async launch:
  build operation from CUdeviceptr
  drop the owning buffer
  run operation later  -> kernel sees stale memory

borrowed typed launch:
  module.kernel_async(..., &input, &mut output)
  Rust keeps those buffers borrowed until the operation is done

owned typed launch:
  module.kernel_async_owned(..., input, output)
  the operation owns the buffers and can be spawned safely
```

Prefer generated `#[cuda_module]` async methods for application code. Use raw
device pointers only when you can prove the allocation outlives every scheduled
operation that may touch it.
