# Error Handling and Debugging

GPU kernels fail differently from CPU code. The CUDA toolchain does not
support exceptions or stack unwinding today, there are no stack traces in
kernel output, and no `println!`. When something goes wrong, the result is
either silent data corruption, a hardware trap, or a cryptic driver error on the
host. This chapter covers cuda-oxide's tools for diagnosing and fixing kernel
problems.

## What happens when a kernel goes wrong

GPU errors fall into three categories:

| Failure mode           | What you see                             | Example                                            |
|:-----------------------|:-----------------------------------------|:---------------------------------------------------|
| **Silent corruption**  | Wrong results, no error                  | Race condition, off-by-one index                   |
| **Hardware trap**      | `CUDA_ERROR_ILLEGAL_INSTRUCTION` on host | `gpu_assert!` failure, panic, OOB access           |
| **Launch failure**     | `DriverError` returned immediately       | Wrong grid dims, missing module, out of resources  |

The CUDA toolchain does not expose an exception mechanism today (the hardware
could support it, but nvcc/ptxas do not wire it up). A trap instruction kills
the kernel and poisons the CUDA context -- subsequent operations on the same
context will fail until you handle or recreate it.

## `gpu_printf!` -- printing from the GPU

`gpu_printf!` lets you print values from device code for quick debugging. It
uses CUDA's built-in `vprintf` mechanism:

```rust
use cuda_device::{kernel, thread, gpu_printf, DisjointSlice};

#[kernel]
pub fn debug_kernel(data: &[f32], mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    if idx.get() < 4 {
        gpu_printf!("Thread {} sees value {}\n", idx.get(), data[idx.get()]);
    }
    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()] * 2.0;
    }
}
```

### Important details

- **Flush requires sync.** Output is buffered on the GPU and only appears on
  the host after a stream or device synchronization (e.g., `to_host_vec` or
  `ctx.synchronize()`).
- **Buffer size.** The default printf buffer is 1 MiB. If many threads print,
  output may be truncated. Enlarge with
  `cudaDeviceSetLimit(cudaLimitPrintfFifoSize, size)`.
- **Thread ordering.** Output from different threads appears in arbitrary order.
- **Performance.** Printf serializes across threads -- avoid it in hot paths.
  Use it for debugging, not logging.
- **Format conversion.** The macro converts Rust `{}` format specifiers to C
  printf equivalents (`%d`, `%f`, etc.) at compile time.

### Why not `println!` or `Debug`?

Standard Rust formatting (`fmt::Display`, `fmt::Debug`, `format!`, `println!`)
requires dynamic dispatch, string allocation, and I/O -- none of which exist on
the GPU. `gpu_printf!` bypasses all of this by lowering directly to a CUDA
`vprintf` call.

## `gpu_assert!` and `trap()`

For fatal error checking on the device, use `gpu_assert!` or `debug::trap()`:

```rust
use cuda_device::{kernel, thread, debug, gpu_assert, DisjointSlice};

#[kernel]
pub fn checked_kernel(data: &[f32], len: u32, mut out: DisjointSlice<f32>) {
    let idx = thread::index_1d();
    gpu_assert!(idx.get() < len as usize);   // traps if false

    if let Some(out_elem) = out.get_mut(idx) {
        *out_elem = data[idx.get()];
    }
}
```

| Intrinsic                | What it does                | Host effect                                    |
|:-------------------------|:----------------------------|:-----------------------------------------------|
| `gpu_assert!(condition)` | Traps if condition is false | `CUDA_ERROR_ILLEGAL_INSTRUCTION`               |
| `debug::trap()`          | Unconditional trap          | `CUDA_ERROR_ILLEGAL_INSTRUCTION`               |
| `debug::breakpoint()`    | Emit `brkpt` instruction    | Pauses in cuda-gdb; crashes without debugger   |

### The trap-and-check pattern

A common workflow for catching device-side errors:

```rust
// Launch kernel
module.vecadd(&stream, config, &a, &b, &mut c).expect("Launch failed");

// Synchronize and check for traps
stream.synchronize().expect("Kernel trapped -- check gpu_assert! conditions");
```

If a `gpu_assert!` fires, synchronization returns an error. The error message
doesn't tell you *which* assertion failed, so use `gpu_printf!` alongside
assertions to narrow down the problem.

## Host-side error handling

### `DriverError`

The synchronous launch path returns
`Result<(), DriverError>`. The `DriverError` wraps a CUDA driver result code:

```rust
match module.vecadd(&stream, config, &a, &b, &mut c) {
    Ok(()) => { /* launched successfully */ }
    Err(e) => eprintln!("Launch failed: {e}"),
}
```

### `DeviceError`

The async path (`{kernel}_async` / `DeviceOperation`) uses `DeviceError`,
which wraps driver errors alongside context and scheduling failures:

```rust
use cuda_async::error::DeviceError;

let result: Result<Vec<f32>, DeviceError> = operation.sync();
```

`DeviceError` variants include `Driver`, `Context`, `KernelCache`, `Scheduling`,
`Launch`, and `Internal`.

### `CudaContext::check_err`

After a series of operations, call `check_err()` on the context to surface any
asynchronous errors that may have been recorded:

```rust
ctx.check_err().expect("Asynchronous GPU error detected");
```

## `cargo oxide debug` -- cuda-gdb integration

`cargo oxide debug` builds your kernel with debug info and launches cuda-gdb:

```bash
cargo oxide debug vecadd          # Standard GDB
cargo oxide debug vecadd --tui    # GDB with TUI
cargo oxide debug vecadd --cgdb   # cgdb front-end
```

### Breakpoint workflow

1. Build with debug: `cargo oxide debug <example>`
2. Set a breakpoint on your kernel: `break vecadd`
3. Run: `run`
4. Inspect threads: `cuda thread`, `cuda block`, `cuda warp`
5. Print variables: `print idx`, `print *c_elem`

For programmatic breakpoints, use `debug::breakpoint()` in your kernel code.
When cuda-gdb hits the `brkpt` instruction, it pauses execution and lets you
inspect the GPU state.

:::{tip}
`debug::breakpoint()` will **crash** the kernel if no debugger is attached.
Guard it with a compile-time flag or only use it during debugging sessions.
:::

## `cargo oxide doctor` -- environment validation

Before debugging kernel failures, verify your environment is correctly set up:

```bash
cargo oxide doctor
```

Doctor checks:

| Check           | What it verifies                                                              |
|:----------------|:------------------------------------------------------------------------------|
| Rust toolchain  | Nightly compiler with required components                                     |
| CUDA toolkit    | `nvcc` found and version compatible                                           |
| libNVVM         | `libnvvm.so` (CUDA Toolkit) loadable -- needed for libdevice math kernels     |
| nvJitLink       | `libnvJitLink.so` (CUDA Toolkit) loadable -- needed for libdevice math kernels|
| libdevice       | `libdevice.10.bc` discoverable -- needed for libdevice math kernels           |
| LLVM            | `llc` (21+) available for PTX generation                                      |
| Codegen backend | `librustc_codegen_cuda.so` found (run `cargo oxide setup` to build it)        |

The libNVVM / nvJitLink / libdevice checks fire only when a kernel calls
CUDA libdevice math (`sin`, `cos`, `exp`, `pow`, `sqrt`, ...). If your
kernel is pure arithmetic, those three failing is harmless. They all ship
with the CUDA Toolkit -- no separate download. If any check fails, doctor
prints the standard install location for that component.

## `cargo oxide pipeline` -- inspecting the compilation

When a kernel produces wrong results but no errors, inspect the compilation
pipeline to see exactly what code was generated:

```bash
cargo oxide pipeline vecadd
```

This prints the full pipeline output:

1. **MIR collection** -- which functions the collector found
2. **`dialect-mir`** -- pliron IR modelling Rust MIR (before and after `mem2reg`)
3. **LLVM dialect** -- pliron IR modelling LLVM IR, provided by `pliron-llvm` (after `mir-lower`)
4. **Textual LLVM IR** -- serialized `.ll` file
5. **Final PTX** -- the generated assembly

### Environment variables

For more targeted inspection:

| Variable                       | Effect                            |
|:-------------------------------|:----------------------------------|
| `CUDA_OXIDE_VERBOSE=1`         | Verbose compiler output           |
| `CUDA_OXIDE_SHOW_RUSTC_MIR=1`  | Dump the rustc MIR before import  |

## Profiling with Nsight Compute

For performance debugging, NVIDIA's **Nsight Compute** (`ncu`) provides
roofline analysis, memory throughput, and occupancy metrics:

```bash
ncu --set full ./target/release/my_example
```

cuda-oxide kernels can emit profiler triggers using
`debug::prof_trigger::<N>()`, which generates a `pmevent` instruction that
Nsight Compute and Nsight Systems can capture for timeline annotation.

:::{seealso}
[Nsight Compute Documentation](https://docs.nvidia.com/nsight-compute/)
for the full profiling toolkit.
:::

## Common pitfalls

| Pitfall                          | Symptom                                    | Fix                                                              |
|:---------------------------------|:-------------------------------------------|:-----------------------------------------------------------------|
| Race condition on output buffer  | Wrong results, non-deterministic           | Use `DisjointSlice` instead of raw `*mut T`                      |
| Missing `sync_threads()`         | Stale shared memory reads                  | Add barrier between writes and reads                             |
| Wrong `shared_mem_bytes`         | `LAUNCH_OUT_OF_RESOURCES` or garbage data  | Match `LaunchConfig` to actual `DynamicSharedArray` usage        |
| Out-of-bounds with raw pointers  | Trap or silent corruption                  | Use `DisjointSlice::get_mut` for bounds checking                 |
| `panic!("message")` in kernel    | Compile error (fmt unavailable)            | Use `gpu_assert!` or `debug::trap()`                             |
| Forgetting to sync after launch  | Host reads stale data                      | Call `to_host_vec`, `stream.synchronize()`, or `.sync()`         |
| PTX built for wrong arch         | `NO_BINARY_FOR_GPU`                        | Rebuild with `cargo oxide build --arch sm_XX`                    |

```{figure} images/debug-workflow.svg
:align: center
:width: 100%

Debugging decision tree: kernel problems fall into three categories (compile
error, runtime trap, silent corruption), each with different diagnostic tools.
Common fixes are shown at the bottom.
```
