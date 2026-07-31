# device_global

Tests ordinary Rust `static mut` values in CUDA global memory, non-zero
immutable Rust static tables, and thin pointers stored inside device-global
initializers.

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

The static-initializer relocation kernel covers pointer values that are part
of another static's evaluated allocation:

```rust
static TARGET_A: u32 = 0x1234_5678;
static TARGET_B: u32 = 0xcafe_babe;
static REFERENCE: &u32 = &TARGET_A;
static REFERENCES: [&u32; 2] = [&TARGET_A, &TARGET_B];
static INTERIOR_REFERENCE: &f32 = &STATIC_WEIGHTS[2][1];
```

rustc stores each pointer as literal addend bytes plus a separate provenance
entry naming the target allocation. cuda-oxide keeps both components through
MIR lowering. The LLVM global uses byte-array fields for literal spans and an
integer-width relocation slot for every pointer. The exporter reconstructs
each slot with `getelementptr`, `addrspacecast`, and `ptrtoint` constant
expressions, so the device linker sees an actual relocation rather than a null
placeholder.

The relocation coverage includes:

- a direct static-to-static reference;
- multiple pointer fields in one initializer;
- two fields sharing one target;
- a second independently materialized target;
- an interior pointer with a non-zero byte addend;
- targets reachable only through another static initializer;
- modern opaque-pointer NVVM IR and legacy LLVM 7 typed-pointer NVVM IR.

The one-past-the-end kernel forms a constant pointer whose byte addend equals
the allocation size. Const eval permits forming, but not dereferencing, such a
pointer, so the translator materializes it; the kernel checks its distance from
the base equals the 32-byte allocation.

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
padding, field offsets, or floating-point payload bits. Pointer-width slots are
excluded from the literal byte image and emitted as provenance-preserving LLVM
constant expressions.

Before emitting an initializer, cuda-oxide proves that its typed field loads
use the same offsets and size as rustc. Relocation records are also checked for
pointer width, alignment, bounds, overlap, target identity, target address
space, and addend bounds. Unsupported layouts fail at compile time instead of
producing a wrong value.

The supported relocation scope is intentionally narrow: thin pointers from one
device static to another device static in global or constant memory, including
zero and non-zero byte addends. Anonymous promoted allocations, functions,
vtables, trait-object metadata, slices and other fat pointers, unsized pointees,
packed or unaligned pointer slots, and relocation targets outside device static
storage remain fail-closed.
