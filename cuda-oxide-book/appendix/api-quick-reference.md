# API Quick Reference

This appendix is a condensed reference for the cuda-oxide device and host APIs.
For full documentation, run `cargo doc --no-deps --open` from the workspace
root.

---

## Attributes and Macros

### Kernel and Device Attributes

```rust
use cuda_device::{kernel, device, launch_bounds, cluster_launch, cooperative_launch};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) { /* ... */ }

#[kernel]
#[launch_bounds(256, 2)]
pub fn tuned_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cluster_launch(4, 1, 1)]
pub fn cluster_kernel(data: &mut [f32]) { /* ... */ }

#[kernel]
#[cooperative_launch]
pub fn grid_sync_kernel(data: &mut [f32]) { /* ... */ }

#[device]
fn helper(x: f32) -> f32 { x * x }
```

| Attribute                                   | Purpose                                                             |
|:--------------------------------------------|:--------------------------------------------------------------------|
| `#[kernel]`                                 | Mark a function as a GPU kernel entry point (`.entry` in PTX)       |
| `#[device]`                                 | Mark a helper function or `extern "C"` block for device compilation |
| `#[launch_bounds(max_threads, min_blocks)]` | Occupancy hints for register allocation                             |
| `#[cluster_launch(x, y, z)]`                | Set compile-time cluster dimensions (Hopper+)                       |
| `#[cooperative_launch]`                     | Launch cooperatively via `#[cuda_module]` (enables `grid::sync()`)  |
| `#[convergent]`                             | Mark as convergent (barrier semantics)                              |
| `#[pure]`                                   | Mark as side-effect free                                            |
| `#[readonly]`                               | Mark as read-only                                                   |

### Output Macros

```rust
use cuda_device::{gpu_printf, gpu_assert};

gpu_printf!("thread %d: val = %f\n", idx as i32, val as f64);
gpu_assert!(val >= 0.0);
```

| Macro                        | Purpose                                              |
|:-----------------------------|:-----------------------------------------------------|
| `gpu_printf!(fmt, args...)`  | Device-side formatted output (lowers to `vprintf`)   |
| `gpu_assert!(condition)`     | Runtime assertion; calls `trap()` on failure         |

---

## Thread Identification

```rust
use cuda_device::thread;

let idx     = thread::index_1d();                            // ThreadIndex<'_, Index1D>
let idx2d   = thread::index_2d::<128>();                     // Option<ThreadIndex<'_, Index2D<128>>>
let idx2d_r = unsafe { thread::index_2d_runtime(stride) };   // Option<ThreadIndex<'_, Runtime2DIndex>>

let tid_x  = thread::threadIdx_x();    // u32
let bid_x  = thread::blockIdx_x();     // u32
let bdim_x = thread::blockDim_x();     // u32
```

| Function                                    | Returns                                          | Description                                                |
|:--------------------------------------------|:-------------------------------------------------|:-----------------------------------------------------------|
| `thread::index_1d()`                        | `ThreadIndex<'_, Index1D>`                       | Unique linear index (1D grids)                             |
| `thread::index_2d::<S>()`                   | `Option<ThreadIndex<'_, Index2D<S>>>`            | Const-stride 2D index; mismatched strides are a type error |
| `unsafe thread::index_2d_runtime(s)`        | `Option<ThreadIndex<'_, Runtime2DIndex>>`        | Runtime-stride 2D index; caller asserts `s` is uniform     |
| `thread::index_2d_row()`                    | `usize`                                          | 2D row index                                               |
| `thread::index_2d_col()`                    | `usize`                                          | 2D column index                                            |
| `thread::threadIdx_{x,y,z}()`               | `u32`                                            | Thread index within block                                  |
| `thread::blockIdx_{x,y,z}()`                | `u32`                                            | Block index within grid                                    |
| `thread::blockDim_{x,y,z}()`                | `u32`                                            | Block dimensions                                           |

`thread::index_2d::<S>()` and `thread::index_2d_runtime(s)` return `None`
when the computed column exceeds the stride — use it to skip the
right-edge tail in non-aligned 2D kernels.

`index_2d::<S>` is the safe default; the const generic encodes the stride
in the witness type so two threads can't mint colliding indices by
passing different strides. `index_2d_runtime` is the escape hatch for
launches whose stride is only known at runtime; the caller takes on the
"every thread used the same stride" obligation by writing `unsafe`. Full
discussion in [The Safety Model](../gpu-safety/the-safety-model.md).

---

## Safe Parallel Writes — DisjointSlice

```rust
use cuda_device::{DisjointSlice, kernel};

#[kernel]
pub fn vecadd(a: &[f32], b: &[f32], mut c: DisjointSlice<f32>) {
    if let Some((c_elem, idx)) = c.get_mut_indexed() {
        let i = idx.get();
        *c_elem = a[i] + b[i];
    }
}
```

| Method                  | Signature                                        | Description                                                          |
|:------------------------|:-------------------------------------------------|:---------------------------------------------------------------------|
| `get_mut_indexed`       | `() -> Option<(&mut T, ThreadIndex<'_, IS>)>`    | One-call form: mints the witness and resolves it. Index1D / Index2D. |
| `get_mut`               | `(ThreadIndex<'_, IS>) -> Option<&mut T>`        | Bounds-checked mutable access from an explicit witness               |
| `get_unchecked_mut`     | `(usize) -> &mut T`                              | Unsafe, unchecked access                                             |
| `len`                   | `() -> usize`                                    | Number of elements                                                   |

`get_mut_indexed` is gated on `IndexSpace: IndexFormula` (impl'd by
`Index1D` and `Index2D<S>`). For `Runtime2DIndex` slices, use the
explicit `unsafe { thread::index_2d_runtime(s) }` + `get_mut(idx)` pair.

---

## Shared Memory

```rust
use cuda_device::{SharedArray, DynamicSharedArray, thread};

#[kernel]
pub fn tiled(data: &[f32], mut out: DisjointSlice<f32>) {
    static mut TILE: SharedArray<f32, 256> = SharedArray::UNINIT;
    let tid = thread::threadIdx_x() as usize;
    unsafe { TILE[tid] = data[thread::index_1d().get()]; }
    thread::sync_threads();
    // ... read from TILE ...
}

#[kernel]
pub fn dynamic(data: &[f32]) {
    static mut BUF: DynamicSharedArray<f32> = DynamicSharedArray::UNINIT;
    // Size set at launch via LaunchConfig::shared_mem_bytes
}
```

| Type                      | Description                                               |
|:--------------------------|:----------------------------------------------------------|
| `SharedArray<T, N>`       | Compile-time sized, block-scoped shared memory            |
| `SharedArray<T, N, 128>`  | With 128-byte alignment (required for TMA destinations)   |
| `DynamicSharedArray<T>`   | Runtime-sized shared memory (set via `LaunchConfig`)      |

Both are `!Sync` — concurrent access requires explicit barriers.

---

## Synchronization

### Block-Level

```rust
thread::sync_threads();   // __syncthreads() equivalent
```

### Managed Barriers (Hopper+)

```rust
use cuda_device::{ManagedBarrier, TmaBarrierHandle, Uninit, Ready};

// Typestate lifecycle: Uninit → Ready → Invalidated
let bar: TmaBarrierHandle<Uninit> = TmaBarrierHandle::from_static(ptr);
let bar: TmaBarrierHandle<Ready> = unsafe { bar.init(thread_count) };
let token = bar.arrive();
bar.wait(token);
unsafe { bar.inval() };
```

| Operation                   | Description                                      |
|:----------------------------|:-------------------------------------------------|
| `.init(count)`              | Initialize barrier with expected arrival count   |
| `.arrive()`                 | Signal arrival, returns `BarrierToken`           |
| `.arrive_expect_tx(bytes)`  | Arrive and set expected TX byte count (for TMA)  |
| `.wait(token)`              | Block until all arrivals + TX complete           |
| `.inval()`                  | Invalidate barrier (cleanup)                     |

---

## Warp Primitives

```rust
use cuda_device::warp;

let lane = warp::lane_id();      // 0–31
let wid  = warp::warp_id();

// Shuffle
let partner = warp::shuffle_xor_f32(val, mask);
let from_above = warp::shuffle_down_f32(val, delta);
let from_below = warp::shuffle_up_f32(val, delta);
let from_lane  = warp::shuffle_f32(val, src_lane);

// i32 variants
let partner_i = warp::shuffle_xor_i32(val, mask);

// Vote
let all_true = warp::all(predicate);
let any_true = warp::any(predicate);
let mask     = warp::ballot(predicate);
let count    = warp::popc(mask);
```

### Shuffle Operations

| Function                              | Description                       |
|:--------------------------------------|:----------------------------------|
| `shuffle_xor_{f32,i32}(val, mask)`    | Exchange with lane `id ^ mask`    |
| `shuffle_down_{f32,i32}(val, delta)`  | Read from lane `id + delta`       |
| `shuffle_up_{f32,i32}(val, delta)`    | Read from lane `id - delta`       |
| `shuffle_{f32,i32}(val, src)`         | Read from specific lane           |

### Vote Operations

| Function       | Returns  | Description                                  |
|:---------------|:---------|:---------------------------------------------|
| `all(pred)`    | `bool`   | True if predicate holds for all lanes        |
| `any(pred)`    | `bool`   | True if predicate holds for any lane         |
| `ballot(pred)` | `u32`    | Bitmask of lanes where predicate is true     |
| `popc(mask)`   | `u32`    | Population count of set bits                 |

---

## Atomics

### Scoped GPU Atomics

```rust
use cuda_device::atomic::{DeviceAtomicU32, AtomicOrdering};

static COUNTER: DeviceAtomicU32 = DeviceAtomicU32::new(0);

// In kernel:
COUNTER.fetch_add(1, AtomicOrdering::Relaxed);
let old = COUNTER.load(AtomicOrdering::Acquire);
```

| Scope                                    | Types                         |
|:-----------------------------------------|:------------------------------|
| `DeviceAtomic{U32,I32,U64,I64,F32,F64}`  | `.gpu` scope                  |
| `BlockAtomic{U32,I32,U64,I64,F32,F64}`   | `.cta` scope                  |
| `SystemAtomic{U32,I32,U64,I64,F32,F64}`  | `.sys` scope (CPU-GPU shared) |

`core::sync::atomic` types (`AtomicU32`, `AtomicBool`, etc.) also compile to
GPU code, defaulting to system scope.

---

## TMA — Tensor Memory Accelerator (Hopper+)

```rust
use cuda_device::tma::TmaDescriptor;
use cuda_device::tma::{cp_async_bulk_tensor_2d_g2s, cp_async_bulk_commit_group};

// Host: build descriptor (128 bytes, opaque)
// Device: issue async bulk copy
cp_async_bulk_tensor_2d_g2s(smem_ptr, &desc, coord_x, coord_y, barrier_ptr);
cp_async_bulk_commit_group();
```

| Function                                      | Description                          |
|:----------------------------------------------|:-------------------------------------|
| `cp_async_bulk_tensor_{1..5}d_g2s(...)`       | Global → shared async bulk copy      |
| `cp_async_bulk_tensor_{1..5}d_s2g(...)`       | Shared → global async bulk copy      |
| `cp_async_bulk_tensor_2d_g2s_multicast(...)`  | Multicast to all CTAs in cluster     |
| `cp_async_bulk_commit_group()`                | Commit outstanding copies            |
| `cp_async_bulk_wait_group(n)`                 | Wait until ≤ n groups remain         |

---

## Cluster Programming (Hopper+)

```rust
use cuda_device::cluster;

let rank = cluster::block_rank();        // This block's rank in the cluster
let size = cluster::cluster_size();      // Number of blocks in cluster
cluster::cluster_sync();                 // Barrier across all cluster blocks

// Distributed Shared Memory
let remote_ptr = cluster::map_shared_rank(local_ptr, target_rank);
let val = cluster::dsmem_read_u32(remote_ptr);
```

---

## Tensor Cores — WGMMA (Hopper, SM 90)

```rust
use cuda_device::wgmma;

wgmma::wgmma_fence();
wgmma::wgmma_commit_group();
wgmma::wgmma_wait_group::<0>();
```

Warpgroup MMA: 4 warps (128 threads) issue matrix multiply-accumulate from
shared memory. Operands described by SMEM descriptors; accumulator in registers.

---

## Tensor Cores — tcgen05 (Blackwell, SM 100+)

```rust
use cuda_device::tcgen05::{TmemGuard, TmemUninit, TmemReady};
use cuda_device::SharedArray;

static mut TMEM_SLOT: SharedArray<u32, 1, 4> = SharedArray::UNINIT;

let guard = TmemGuard::<TmemUninit, 512>::from_static(&raw mut TMEM_SLOT as *mut u32);
let guard = unsafe { guard.alloc() };   // TmemUninit → TmemReady
// ... issue MMA, read results via guard.address() ...
let _guard = unsafe { guard.dealloc() }; // TmemReady → TmemDeallocated
```

Single-thread MMA issue into dedicated Tensor Memory (TMEM). `TmemGuard`
manages TMEM lifetime with typestate: `TmemUninit → TmemReady → TmemDeallocated`.
N_COLS must be a power of 2 in the range [32, 512].

---

## Host-Side: Kernel Launch

### Typed Synchronous

```rust
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};

let ctx = CudaContext::new(0).unwrap();
let stream = ctx.default_stream();
let module = kernels::load(&ctx).unwrap();

let a = DeviceBuffer::from_host(&stream, &a_host).unwrap();
let b = DeviceBuffer::from_host(&stream, &b_host).unwrap();
let mut output = DeviceBuffer::<f32>::zeroed(&stream, n).unwrap();

module.vecadd(&stream, LaunchConfig::for_num_elems(n), &a, &b, &mut output).unwrap();
```

### Typed Async

```rust
use cuda_async::device_operation::DeviceOperation;

let module = kernels::load_async(0)?;
let op = module.vecadd_async(LaunchConfig::for_num_elems(n), &a, &b, &mut output)?;

op.sync()?;       // blocking
// or: op.await?;  // async with tokio
```

`cuda_launch!` and `cuda_launch_async!` remain available as lower-level APIs for
explicit module loading and custom launch code.

### LaunchConfig

| Method                                                   | Description                                  |
|:---------------------------------------------------------|:---------------------------------------------|
| `LaunchConfig::for_num_elems(n)`                         | Auto-configure grid/block for `n` elements   |
| `LaunchConfig { grid_dim, block_dim, shared_mem_bytes }` | Direct struct construction                   |

---

## Debug Facilities

```rust
use cuda_device::debug;

let t = debug::clock64();       // Cycle counter
debug::trap();                  // Abort kernel
debug::breakpoint();            // cuda-gdb breakpoint
cuda_device::barrier::nanosleep(1000); // Sleep ~1μs
debug::prof_trigger::<7>();     // Nsight profiler trigger
```

---

## Quick Reference Tables

### cuda-device Modules

| Module               | Description                                                      | Min SM   |
|:---------------------|:-----------------------------------------------------------------|:---------|
| `thread`             | Thread/block IDs, `index_1d`, `sync_threads`                     | All      |
| `disjoint`           | `DisjointSlice<T>` — safe parallel writes                        | All      |
| `shared`             | `SharedArray<T, N>`, `DynamicSharedArray<T>`                     | All      |
| `warp`               | Shuffle, vote, match, lane/warp ID                               | All      |
| `atomic`             | Scoped atomics (device/block/system)                             | sm_70+   |
| `debug`              | `clock64`, `trap`, `breakpoint`, `gpu_printf!`                   | All      |
| `fence`              | `threadfence_block` / `threadfence` / `threadfence_system`       | All      |
| `grid`               | Grid-scoped `sync` (cooperative kernel launches)                 | sm_70+   |
| `cooperative_groups` | Typed handles, warp/block reductions and scans                   | All      |
| `barrier`            | `ManagedBarrier` — async mbarrier for TMA/MMA                    | sm_90+   |
| `cluster`            | Thread block clusters, DSMEM                                     | sm_90+   |
| `tma`                | `TmaDescriptor`, bulk tensor copies (1D–5D)                      | sm_90+   |
| `wgmma`              | Warpgroup MMA (fence/commit/wait)                                | sm_90    |
| `tcgen05`            | 5th-gen tensor cores, TMEM, `TmemGuard`                          | sm_100+  |
| `cusimd`             | `CuSimd<T, N>`, `Float2`/`Float4`                                | All      |
| `clc`                | Cluster Launch Control                                           | sm_100+  |

### Crate Map

| Crate             | Role                                                                   |
|:------------------|:-----------------------------------------------------------------------|
| `cuda-device`     | Device intrinsics and types (`#![no_std]`)                             |
| `cuda-macros`     | Proc macros (`#[kernel]`, `#[device]`, `gpu_printf!`)                  |
| `cuda-host`       | Typed module loading plus low-level launch helpers                     |
| `cuda-core`       | Safe RAII wrappers (`CudaContext`, `CudaStream`, `DeviceBuffer<T>`)    |
| `cuda-async`      | `DeviceOperation`, `DeviceFuture`, `DeviceBox<T>`                      |
| `cuda-bindings`   | Raw `bindgen` FFI to `cuda.h`                                          |
| `cargo-oxide`     | Cargo subcommand (`cargo oxide run`, `build`, `debug`)                 |
