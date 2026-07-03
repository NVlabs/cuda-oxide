# standalone_device_fn

Standalone `#[device]` function compilation — no `#[kernel]` required.

## What This Demonstrates

This example verifies that `#[device]` functions can be compiled to PTX without
any `#[kernel]` entry point in the crate. This is the foundation for building
Rust device libraries that are consumed by CUDA C++ via LTOIR linking.

## Test Coverage

| # | Test                       | What It Verifies                                                         |
|---|----------------------------|--------------------------------------------------------------------------|
| 1 | Simple device functions    | `fast_sqrt`, `clamp_f32` appear as `.func` in PTX                        |
| 2 | Transitive calls           | `safe_sqrt` (calls `fast_sqrt` + `clamp_f32`) is collected               |
| 3 | Generic device function    | `fma_f32` (concrete wrapper around generic `fma<T>`) compiles            |
| 4 | GPU intrinsics             | `get_global_thread_id` (uses `thread::index_1d()`) compiles              |
| 5 | Multiple monomorphizations | Both `fma_f32` and `fma_i32` instantiations present                      |
| 6 | Uninstantiated generic     | Generic `lerp<T>` is correctly **absent** (not monomorphized)            |
| 7 | No `.entry` directives     | All functions are `.func`, not `.entry` (no kernels)                     |
| 8 | INT8 MMA compiler path     | Device stub imports and lowers to the exact `mma.sync` PTX instruction   |
| 9 | INT8 MMA target floors     | Generated PTX selects PTX 7.0 / `sm_80` or newer                         |

## How to Run

```bash
# From workspace root
cargo oxide run standalone_device_fn
```

Expected output:

```text
=== Standalone Device Function Example ===

PTX file: standalone_device_fn.ptx (... bytes)

  PASS  fast_sqrt — Test 1: simple standalone fn
  PASS  clamp_f32 — Test 1: simple standalone fn
  PASS  safe_sqrt — Test 2: device fn calling device fn
  PASS  fma_f32 — Test 3: generic instantiation (f32)
  PASS  fma_i32 — Test 3: generic instantiation (i32)
  PASS  get_global_thread_id — Test 4: device fn with GPU intrinsics
  PASS  int8_mma_registers — Test 5: cuda-device stub imported and lowered
  PASS  lerp absent — Test 3b: uninstantiated generic not compiled

  PASS  No .entry directives (all are .func)
  PASS  INT8 MMA lowered to the exact PTX instruction
  PASS  INT8 MMA selected PTX 7.0 or newer
  PASS  INT8 MMA selected sm_80 or newer

SUCCESS: 12/12 tests passed — all device functions compiled to PTX!
```

## How It Works

1. The `#[device]` macro renames functions with the reserved
   `cuda_oxide_device_<hash>_` prefix (owned by
   `crates/reserved-oxide-symbols/`) and generates an `#[inline(always)]`
   wrapper with the original name
2. The collector (`rustc-codegen-cuda/src/collector.rs`) detects standalone
   `#[device]` functions as compilation roots when no `#[kernel]` is present
3. The LLVM export layer strips the `cuda_oxide_device_<hash>_` prefix for clean
   names in the final output
4. Generic `#[device]` functions are only compiled if monomorphized by a
   concrete call site (standard Rust monomorphization rules)

## Related

- `cpp_consumes_rust_device/` — Takes this further: compiles to LTOIR and links with C++
