# device_global

Tests ordinary Rust `static mut` values in CUDA global memory and non-zero
immutable Rust static tables.

Run with:

```bash
cargo oxide run device_global
```

The first kernel updates two ordinary device statics:

```rust
static mut DEVICE_COUNTER: u64 = 0;
static mut DEVICE_MARKER: u32 = 0;
```

The other kernels read non-zero immutable statics. One reads both the base
address and an interior constant pointer (`&STATIC_WEIGHTS[2]`, a 16-byte
addend), matching generated coefficient-table access patterns:

```rust
static STATIC_WEIGHTS: [[f32; 2]; 4] = [[0.25, 0.5], ...];
const STATIC_WEIGHT_PAIR: &[f32; 2] = &STATIC_WEIGHTS[2];
```

The subobject kernel takes references to fields and indexed elements of
statics (`&PADDED_STATIC.tag`, `&PADDED_STATIC.value`, `&STATIC_WEIGHTS[2]`).
The address arithmetic stays in the static's physical address space and a
single address-space cast restores the generic Rust pointer type at the
borrow boundary.

The one-past-the-end kernel forms a constant pointer whose byte addend
equals the allocation size. Const eval permits forming (not dereferencing)
such a pointer, so the translator materializes it; the kernel checks its
distance from the base equals the 32-byte allocation.

The edge-case kernel checks two byte-level details:

- `STATIC_NAN` keeps the complete `0x7fc01234` NaN payload instead of being
  rewritten to a canonical NaN.
- `PADDED_STATIC`, a `#[repr(C)] { u8, u32 }`, reads the `u32` from its Rust
  layout offset after three padding bytes.

Expected behavior:

| Static kind                  | Memory space          |
|------------------------------|-----------------------|
| Ordinary `static mut`        | Global `addrspace(1)` |
| `SharedArray` / `Barrier`    | Shared `addrspace(3)` |
| `DynamicSharedArray::get()`  | Shared `addrspace(3)` |

The example launches the kernel twice. `DEVICE_COUNTER` should persist across
launches, proving it is global device storage and not per-block shared memory.

Non-zero immutable static initializers are emitted as the exact evaluated byte
image in LLVM/PTX, so device code can read compile-time data without losing
padding, field offsets, or floating-point payload bits.

Before emitting those bytes, cuda-oxide also proves that its typed field loads
use the same offsets and size as rustc. Layouts that are not modeled exactly
fail at compile time instead of producing a wrong value. This currently
includes packed structs, non-empty tuples, niche-encoded enums, unions, and
constant pointers into a static whose pointee is unsized (a slice or another
DST needing fat-pointer metadata). Initializers that contain pointer
relocations remain unsupported for the same reason: replacing a relocation
with literal zero bytes would silently turn the pointer into null.
