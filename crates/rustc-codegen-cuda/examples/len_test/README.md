# len_test

## Rvalue::Len - Slice and Array Length on Device

This example demonstrates calling `.len()` on a device-side slice, exercising
the `Rvalue::Len` codegen path in `mir-importer`. Each thread reads the length
of the shared input slice and writes it to its own output slot.

## What This Example Does

- Allocates an input slice of 1024 `f32` values and an output buffer of 1024 `u64` values
- Launches a kernel where each thread computes `input.len()` and writes it to `out[idx]`
- Verifies every output element equals the true input length on the host

## Key Concepts Demonstrated

### Slice Length on the Device

```rust
#[kernel]
pub fn len_test(input: &[f32], mut out: DisjointSlice<u64>) {
    let idx = thread::index_1d();
    let len = input.len() as u64;
    if let Some(o) = out.get_mut(idx) {
        *o = len;
    }
}
```

`input.len()` compiles to `mir.extract_field` reading the length field of the
slice's `{ ptr, len }` representation (see `MirSliceType`). For a fixed-size
array `[T; N]` instead of a slice, the length is a compile-time constant taken
from `MirArrayType::size()` and emitted as a `mir.constant`, with no runtime
read at all.

### Thread Indexing

- `thread::index_1d()` mints the per-thread `ThreadIndex`.
- `out.get_mut(idx)` resolves it to a `&mut u64`, returning `None` for
  out-of-bounds threads.

## Build and Run

```bash
cargo oxide run len_test
```

## Expected Output

```text
=== Rvalue::Len Test ===
Input slice length: 1024
Output (first 5 elements): [1024, 1024, 1024, 1024, 1024]
✓ SUCCESS: All 1024 elements report correct slice length!
```

## Hardware Requirements

- **Minimum GPU**: Any CUDA-capable GPU (Kepler or newer recommended)
- **CUDA Driver**: 11.0+

## Potential Errors

| Error                                | Cause                      | Solution                                    |
|---------------------------------------|----------------------------|------------------------------------------------|
| `CUDA_ERROR_NO_DEVICE`                | No GPU found               | Ensure NVIDIA driver is installed             |
| `Failed to load embedded CUDA module` | Embedded PTX was not found | Build through `cargo oxide run len_test`      |
| `Kernel launch failed`                | Invalid launch config      | Ensure grid/block dims don't exceed limits    |

## How It Works Under the Hood

1. **rustc** parses the file, generates MIR for the kernel, including an
   `Rvalue::Len(place)` node for the `.len()` call
2. **mir-importer** translates `Rvalue::Len`:
   - Array place → emits a `mir.constant` with the type's static size
   - Slice place → emits `mir.extract_field` reading field 1 of the
     `{ ptr, len }` pair
3. **rustc-codegen-cuda** routes the kernel to PTX and `main` to standard LLVM